package main

import (
	"math"
	"sort"
)

// CandidateGranularity holds a granularity and its estimated latency
type CandidateGranularity struct {
	W, H, K  int
	Latency  float64
	WorkSet  int64
	Feasible bool
}

// GenerateCandidateGranularitiesSmart produces a focused set of [w,h,k] to try
// Uses heuristics to avoid exponential search space
func GenerateCandidateGranularitiesSmart(p *Problem, ops []int, residentTensors map[int]bool) []CandidateGranularity {
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]

	primaryOutput := GetOutputTensor(p, ops)
	outT := p.Tensors[primaryOutput]

	maxK := GetMaxK(p, ops)

	// Smart candidate generation: focus on native and powers of 2
	wCandidates := generateSmartDimensionCandidates(nw, outT.Width, 8) // max 8 candidates
	hCandidates := generateSmartDimensionCandidates(nh, outT.Height, 8)
	kCandidates := generateKCandidates(maxK, 6) // max 6 k values

	var candidates []CandidateGranularity

	// Prioritize native granularity first
	priorityGrans := [][3]int{
		{nw, nh, maxK},             // Native with full K
		{nw, nh, MinInt(maxK, nw)}, // Native with reasonable K
		{nw / 2, nh, maxK},         // Half-width
		{nw, nh / 2, maxK},         // Half-height
	}

	for _, gran := range priorityGrans {
		w, h, k := gran[0], gran[1], gran[2]
		if w > 0 && h > 0 && k > 0 && w <= outT.Width && h <= outT.Height && k <= maxK {
			ws := ComputeWorkingSet(p, ops, [3]int{w, h, k}, residentTensors)
			feasible := ws <= p.FastMemoryCapacity
			lat := math.Inf(1)
			if feasible {
				lat = EvaluateSubgraphSimple(p, ops, [3]int{w, h, k}, nil, residentTensors)
			}
			candidates = append(candidates, CandidateGranularity{
				W: w, H: h, K: k,
				Latency:  lat,
				WorkSet:  ws,
				Feasible: feasible,
			})
		}
	}

	// Then try a focused grid search
	for _, w := range wCandidates {
		for _, h := range hCandidates {
			for _, k := range kCandidates {
				gran := [3]int{w, h, k}
				ws := ComputeWorkingSet(p, ops, gran, residentTensors)

				feasible := ws <= p.FastMemoryCapacity
				lat := math.Inf(1)
				if feasible {
					lat = EvaluateSubgraphSimple(p, ops, gran, nil, residentTensors)
				}

				candidates = append(candidates, CandidateGranularity{
					W: w, H: h, K: k,
					Latency:  lat,
					WorkSet:  ws,
					Feasible: feasible,
				})
			}
		}
	}

	return candidates
}

// generateSmartDimensionCandidates creates a focused set of tile sizes
func generateSmartDimensionCandidates(native, tensorSize, maxCount int) []int {
	candidates := make(map[int]bool)

	// Always include native
	candidates[native] = true

	// Include tensor size if reasonable
	if tensorSize <= native*4 {
		candidates[tensorSize] = true
	}

	// Powers of 2 around native
	for v := native; v >= native/8 && v >= 16; v /= 2 {
		if v <= tensorSize {
			candidates[v] = true
		}
	}

	for v := native * 2; v <= tensorSize && v <= native*4; v *= 2 {
		candidates[v] = true
	}

	// Include some multiples of native
	if native*2 <= tensorSize {
		candidates[native*2] = true
	}

	result := make([]int, 0, len(candidates))
	for v := range candidates {
		result = append(result, v)
	}
	sort.Ints(result)

	// Limit to maxCount
	if len(result) > maxCount {
		// Keep the ones closest to native
		sort.Slice(result, func(i, j int) bool {
			di := AbsInt(result[i] - native)
			dj := AbsInt(result[j] - native)
			return di < dj
		})
		result = result[:maxCount]
		sort.Ints(result)
	}

	return result
}

// generateKCandidates creates a focused set of k values
func generateKCandidates(maxK, maxCount int) []int {
	if maxK <= 1 {
		return []int{1}
	}

	candidates := []int{maxK} // Always include full K

	// Powers of 2 down from maxK
	for k := maxK / 2; k >= 1; k /= 2 {
		candidates = append(candidates, k)
	}

	// Also include some strategic values
	if maxK >= 256 {
		candidates = append(candidates, 256)
	}
	if maxK >= 128 {
		candidates = append(candidates, 128)
	}
	if maxK >= 64 {
		candidates = append(candidates, 64)
	}

	// Deduplicate and sort
	candidateMap := make(map[int]bool)
	for _, k := range candidates {
		candidateMap[k] = true
	}

	result := make([]int, 0, len(candidateMap))
	for k := range candidateMap {
		result = append(result, k)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(result)))

	if len(result) > maxCount {
		result = result[:maxCount]
	}

	return result
}

// FindBestGranularity returns the optimal granularity for a subgraph
func FindBestGranularity(p *Problem, ops []int, residentTensors map[int]bool) [3]int {
	candidates := GenerateCandidateGranularitiesSmart(p, ops, residentTensors)

	bestLat := math.Inf(1)
	bestGran := [3]int{1, 1, 1}

	for _, cand := range candidates {
		if cand.Feasible && cand.Latency < bestLat {
			bestLat = cand.Latency
			bestGran = [3]int{cand.W, cand.H, cand.K}
		}
	}

	// Fallback if nothing feasible
	if math.IsInf(bestLat, 1) {
		bestGran = findSmallestFeasibleGranularity(p, ops, residentTensors)
	}

	return bestGran
}

// findSmallestFeasibleGranularity finds the smallest [w,h,k] that fits
func findSmallestFeasibleGranularity(p *Problem, ops []int, residentTensors map[int]bool) [3]int {
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]
	maxK := GetMaxK(p, ops)

	// Start from native and halve until it fits
	for w := nw; w >= 1; w /= 2 {
		for h := nh; h >= 1; h /= 2 {
			for k := maxK; k >= 1; k /= 2 {
				gran := [3]int{w, h, k}
				ws := ComputeWorkingSet(p, ops, gran, residentTensors)
				if ws <= p.FastMemoryCapacity {
					return gran
				}
			}
		}
	}

	return [3]int{1, 1, 1}
}

// IsGranularityReasonable checks if a granularity is acceptable
// Returns false if tiles are too small relative to native (too much padding overhead)
func IsGranularityReasonable(p *Problem, gran [3]int) bool {
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]
	w, h := gran[0], gran[1]

	// Don't allow tiles smaller than 1/4 of native in any dimension
	// This avoids excessive padding overhead
	if w < nw/4 || h < nh/4 {
		return false
	}

	return true
}
