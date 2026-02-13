package main

import (
	"math"
	"sort"
)

// TryFuseOps checks if fusing a set of ops is feasible and beneficial
func TryFuseOps(p *Problem, ops []int, residentTensors map[int]bool) (feasible bool, gran [3]int, lat float64) {
	gran = FindBestGranularity(p, ops, residentTensors)
	ws := ComputeWorkingSet(p, ops, gran, residentTensors)

	if ws > p.FastMemoryCapacity {
		return false, gran, math.Inf(1)
	}

	trav := BestTraversal(p, ops, gran)
	lat, err := EvaluateSubgraphDetailed(p, ops, gran, nil, trav, residentTensors)
	if err != nil {
		return false, gran, math.Inf(1)
	}

	return true, gran, lat
}

// EstimateUnfusedLatency estimates cost of running ops individually in sequence
func EstimateUnfusedLatency(p *Problem, ops []int, residentTensors map[int]bool) float64 {
	total := 0.0
	bw := float64(p.SlowMemoryBandwidth)

	for i, opIdx := range ops {
		singleOps := []int{opIdx}
		gran := FindBestGranularity(p, singleOps, residentTensors)
		trav := BestTraversal(p, singleOps, gran)
		lat, err := EvaluateSubgraphDetailed(p, singleOps, gran, nil, trav, residentTensors)
		if err != nil {
			lat = math.Inf(1)
		}
		total += lat

		// Add intermediate transfer cost (evict + reload)
		if i < len(ops)-1 {
			op := p.Ops[opIdx]
			for _, outT := range op.Outputs {
				size := FullTensorSize(p, outT)
				total += 2.0 * float64(size) / bw
			}
		}
	}

	return total
}

// FuseChainDP uses dynamic programming to find the best fusion of a chain
func FuseChainDP(p *Problem, chain []int, residentTensors map[int]bool) [][]int {
	n := len(chain)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return [][]int{chain}
	}

	maxSegLen := MinInt(n, 6)

	dp := make([]float64, n+1)
	split := make([]int, n+1)

	dp[0] = 0
	for i := 1; i <= n; i++ {
		dp[i] = math.Inf(1)
		for j := MaxInt(0, i-maxSegLen); j < i; j++ {
			segment := chain[j:i]
			segResident := residentTensors
			if j > 0 {
				segResident = make(map[int]bool)
			}

			feasible, _, segLat := TryFuseOps(p, segment, segResident)

			if !feasible {
				segLat = EstimateUnfusedLatency(p, segment, segResident)
			}

			transferCost := 0.0
			if j > 0 {
				boundary := GetSubgraphBoundary(p, segment)
				bw := float64(p.SlowMemoryBandwidth)
				for tIdx := range boundary.BoundaryInputs {
					for _, prevOp := range chain[:j] {
						for _, outT := range p.Ops[prevOp].Outputs {
							if outT == tIdx {
								transferCost += 2.0 * float64(FullTensorSize(p, tIdx)) / bw
							}
						}
					}
				}
			}

			cost := dp[j] + segLat + transferCost
			if cost < dp[i] {
				dp[i] = cost
				split[i] = j
			}
		}
	}

	var segments [][]int
	i := n
	for i > 0 {
		j := split[i]
		segments = append([][]int{chain[j:i]}, segments...)
		i = j
	}

	return segments
}

// FuseChainGreedy uses a greedy approach to fuse consecutive ops
func FuseChainGreedy(p *Problem, chain []int, residentTensors map[int]bool) [][]int {
	if len(chain) <= 1 {
		return [][]int{chain}
	}

	var groups [][]int
	currentGroup := []int{chain[0]}

	for i := 1; i < len(chain); i++ {
		candidate := append(append([]int{}, currentGroup...), chain[i])
		feasible, _, fusedLat := TryFuseOps(p, candidate, residentTensors)

		if !feasible {
			groups = append(groups, currentGroup)
			currentGroup = []int{chain[i]}
			continue
		}

		separateLat := 0.0
		_, _, currentLat := TryFuseOps(p, currentGroup, residentTensors)
		_, _, nextLat := TryFuseOps(p, []int{chain[i]}, make(map[int]bool))
		separateLat = currentLat + nextLat

		bw := float64(p.SlowMemoryBandwidth)
		prevOp := p.Ops[chain[i-1]]
		for _, outT := range prevOp.Outputs {
			separateLat += 2.0 * float64(FullTensorSize(p, outT)) / bw
		}

		if fusedLat < separateLat {
			currentGroup = candidate
		} else {
			groups = append(groups, currentGroup)
			currentGroup = []int{chain[i]}
		}
	}

	groups = append(groups, currentGroup)
	return groups
}

// tryCrossChainFusion tries to fuse groups that share large inputs
func tryCrossChainFusion(p *Problem, gi *GraphInfo, groups [][]int) [][]int {
	if len(groups) <= 1 {
		return groups
	}

	tensorToGroups := make(map[int][]int)
	for gIdx, group := range groups {
		boundary := GetSubgraphBoundary(p, group)
		for tIdx := range boundary.BoundaryInputs {
			tensorToGroups[tIdx] = append(tensorToGroups[tIdx], gIdx)
		}
	}

	type fusionCandidate struct {
		g1, g2   int
		sharedBW int64
	}

	var candidates []fusionCandidate

	for tIdx, gIdxs := range tensorToGroups {
		if len(gIdxs) < 2 {
			continue
		}
		tSize := FullTensorSize(p, tIdx)
		if tSize < 1024 {
			continue
		}

		for i := 0; i < len(gIdxs); i++ {
			for j := i + 1; j < len(gIdxs); j++ {
				g1, g2 := gIdxs[i], gIdxs[j]
				if canFuseGroups(p, gi, groups[g1], groups[g2]) {
					candidates = append(candidates, fusionCandidate{g1, g2, tSize})
				}
			}
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].sharedBW > candidates[j].sharedBW
	})

	merged := make(map[int]bool)

	// Pre-calculate baselines
	baselines := make(map[int]struct {
		lat  float64
		gran [3]int
	})

	for i, grp := range groups {
		_, g, l := TryFuseOps(p, grp, make(map[int]bool))
		baselines[i] = struct {
			lat  float64
			gran [3]int
		}{l, g}
	}

	for _, cand := range candidates {
		g1, g2 := cand.g1, cand.g2
		if merged[g1] || merged[g2] {
			continue
		}

		// Constraint: Don't fuse huge number of disjoint ops
		if len(groups[g1])+len(groups[g2]) > 8 {
			continue
		}

		// Constraint: STRICTLY avoid cross-fusing heavy compute operations.
		// If operations have high base cost (like massive MatMuls), fusing them
		// usually hurts because it constrains the tiling grid for both, reducing K-dimension efficiency.
		isHeavy := false
		for _, opIdx := range groups[g1] {
			if p.Ops[opIdx].BaseCost > 2000 {
				isHeavy = true
				break
			}
		}
		if !isHeavy {
			for _, opIdx := range groups[g2] {
				if p.Ops[opIdx].BaseCost > 2000 {
					isHeavy = true
					break
				}
			}
		}
		if isHeavy {
			continue
		}

		combined := append(append([]int{}, groups[g1]...), groups[g2]...)

		if !isTopologicallyValid(p, gi, combined) {
			continue
		}

		combined = sortOpsTopologically(gi, combined)

		feasible, fusedGran, fusedLat := TryFuseOps(p, combined, make(map[int]bool))
		if !feasible {
			continue
		}

		// Constraint: PADDING CHECK
		outT := p.Tensors[GetOutputTensor(p, combined)]
		if float64(fusedGran[0]) > float64(outT.Width)*1.05 {
			continue
		}
		if float64(fusedGran[1]) > float64(outT.Height)*1.05 {
			continue
		}

		// Constraint: K-DIMENSION PRESERVATION
		base1 := baselines[g1]
		base2 := baselines[g2]
		targetK := 0
		if HasMatMul(p, groups[g1]) {
			targetK = base1.gran[2]
		}
		if HasMatMul(p, groups[g2]) {
			if targetK == 0 || base2.gran[2] < targetK {
				targetK = base2.gran[2]
			}
		}
		if targetK > 1 {
			if float64(fusedGran[2]) < float64(targetK)*0.5 {
				continue
			}
		}

		separateLat := base1.lat + base2.lat
		bw := float64(p.SlowMemoryBandwidth)
		transferCost := 0.0

		b1 := GetSubgraphBoundary(p, groups[g1])
		b2 := GetSubgraphBoundary(p, groups[g2])
		for tIdx := range b1.BoundaryOutputs {
			if b2.BoundaryInputs[tIdx] {
				transferCost += 2.0 * float64(FullTensorSize(p, tIdx)) / bw
			}
		}
		for tIdx := range b2.BoundaryOutputs {
			if b1.BoundaryInputs[tIdx] {
				transferCost += 2.0 * float64(FullTensorSize(p, tIdx)) / bw
			}
		}

		separateLat += transferCost

		if fusedLat < separateLat*0.90 { // Strict requirement for improvement
			groups[g1] = combined
			merged[g2] = true

			baselines[g1] = struct {
				lat  float64
				gran [3]int
			}{fusedLat, fusedGran}
		}
	}

	var result [][]int
	for i, group := range groups {
		if !merged[i] {
			result = append(result, group)
		}
	}

	return result
}
