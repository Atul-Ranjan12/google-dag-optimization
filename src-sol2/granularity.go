package main

import (
	"math"
	"sort"
)

type CandidateGranularity struct {
	W, H, K  int
	Latency  float64
	WorkSet  int64
	Feasible bool
}

func FindBestGranularity(p *Problem, ops []int, residentTensors map[int]bool) [3]int {
	candidates := generateCandidates(p, ops, residentTensors)

	bestLat := math.Inf(1)
	bestGran := [3]int{1, 1, 1}

	for _, c := range candidates {
		if c.Feasible && c.Latency < bestLat {
			bestLat = c.Latency
			bestGran = [3]int{c.W, c.H, c.K}
		}
	}

	if math.IsInf(bestLat, 1) {
		bestGran = findSmallestFeasible(p, ops, residentTensors)
	}

	return bestGran
}

func FindBestGranularityWithRetain(p *Problem, ops []int, residentTensors map[int]bool, retainAfter []int) [3]int {
	candidates := generateCandidates(p, ops, residentTensors)

	bestLat := math.Inf(1)
	bestGran := [3]int{1, 1, 1}

	for _, c := range candidates {
		if !c.Feasible {
			continue
		}
		wsRetain := ComputeWorkingSetWithRetained(p, ops, [3]int{c.W, c.H, c.K}, residentTensors, retainAfter)
		if wsRetain > p.FastMemoryCapacity {
			continue
		}
		if c.Latency < bestLat {
			bestLat = c.Latency
			bestGran = [3]int{c.W, c.H, c.K}
		}
	}

	if math.IsInf(bestLat, 1) {
		bestGran = findSmallestFeasible(p, ops, residentTensors)
	}

	return bestGran
}

func generateCandidates(p *Problem, ops []int, residentTensors map[int]bool) []CandidateGranularity {
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]
	primaryOutput := GetOutputTensor(p, ops)
	outT := p.Tensors[primaryOutput]
	maxK := GetMaxK(p, ops)
	hasMatmul := HasMatMul(p, ops)

	wCands := generateDimCandidates(nw, outT.Width)
	hCands := generateDimCandidates(nh, outT.Height)

	if !containsInt(wCands, outT.Width) {
		wCands = append(wCands, outT.Width)
	}
	if !containsInt(hCands, outT.Height) {
		hCands = append(hCands, outT.Height)
	}

	var kCands []int
	if hasMatmul {
		kCands = generateKCandidates(maxK)
	} else {
		kCands = []int{1}
	}

	capCands := capacityDrivenCandidates(p, ops, residentTensors, outT.Width, outT.Height, maxK, hasMatmul)

	var candidates []CandidateGranularity
	evaluated := make(map[[3]int]bool)

	addCandidate := func(w, h, k int) {
		if w <= 0 || h <= 0 || k <= 0 {
			return
		}
		if w > outT.Width {
			w = outT.Width
		}
		if h > outT.Height {
			h = outT.Height
		}
		if hasMatmul && k > maxK {
			k = maxK
		}

		key := [3]int{w, h, k}
		if evaluated[key] {
			return
		}
		evaluated[key] = true

		gran := [3]int{w, h, k}
		ws := ComputeWorkingSet(p, ops, gran, residentTensors)
		feasible := ws <= p.FastMemoryCapacity
		lat := math.Inf(1)
		if feasible {
			lat = QuickEstimate(p, ops, gran, residentTensors)
		}
		candidates = append(candidates, CandidateGranularity{
			W: w, H: h, K: k, Latency: lat, WorkSet: ws, Feasible: feasible,
		})
	}

	for _, w := range wCands {
		for _, h := range hCands {
			for _, k := range kCands {
				addCandidate(w, h, k)
			}
		}
	}

	for _, c := range capCands {
		addCandidate(c[0], c[1], c[2])
	}

	addCandidate(nw, nh, maxK)
	addCandidate(nw, nh, 1)

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Feasible != candidates[j].Feasible {
			return candidates[i].Feasible
		}

		// Optimization: Check K first (Desc) for MatMul
		// Maximizing K (reduction depth) minimizes output stationarity overhead
		if candidates[i].K != candidates[j].K {
			return candidates[i].K > candidates[j].K
		}

		// Tolerance for floating point equality
		diff := candidates[i].Latency - candidates[j].Latency
		if math.Abs(diff) > 1.0 {
			return diff < 0
		}

		// Break ties with area
		areaI := candidates[i].W * candidates[i].H
		areaJ := candidates[j].W * candidates[j].H
		return areaI > areaJ
	})

	// Refine top N with detailed evaluation
	topN := MinInt(20, len(candidates))
	for i := 0; i < topN; i++ {
		c := &candidates[i]
		if !c.Feasible {
			continue
		}
		gran := [3]int{c.W, c.H, c.K}
		outT := p.Tensors[GetOutputTensor(p, ops)]
		nCols := CeilDiv(outT.Width, c.W)
		nRows := CeilDiv(outT.Height, c.H)
		var trav []int
		if HasMatMul(p, ops) && nCols*nRows > 1 {
			trav = SnakeTraversal(nCols, nRows)
		}
		lat, err := EvaluateSubgraphDetailed(p, ops, gran, nil, trav, residentTensors)
		if err == nil {
			c.Latency = lat
		}
	}

	return candidates
}

func generateDimCandidates(native, tensorSize int) []int {
	cands := make(map[int]bool)
	cands[native] = true

	for v := native * 2; v <= tensorSize; v *= 2 {
		cands[v] = true
	}
	for v := native / 2; v >= MaxInt(native/8, 1); v /= 2 {
		cands[v] = true
	}

	if tensorSize > 0 {
		cands[tensorSize] = true
	}
	if native > 0 && tensorSize%native == 0 {
		cands[tensorSize] = true
	}

	var result []int
	for v := range cands {
		if v > 0 && v <= tensorSize {
			result = append(result, v)
		}
	}
	sort.Ints(result)
	return result
}

func generateKCandidates(maxK int) []int {
	if maxK <= 1 {
		return []int{1}
	}
	cands := make(map[int]bool)
	cands[maxK] = true

	for k := maxK / 2; k >= 1; k /= 2 {
		cands[k] = true
	}
	for _, v := range []int{32, 64, 128, 256, 512, 1024} {
		if v <= maxK {
			cands[v] = true
		}
	}

	var result []int
	for k := range cands {
		result = append(result, k)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(result)))
	if len(result) > 8 {
		result = result[:8]
	}
	return result
}

func capacityDrivenCandidates(p *Problem, ops []int, residentTensors map[int]bool,
	outW, outH, maxK int, hasMatmul bool) [][3]int {

	boundary := GetSubgraphBoundary(p, ops)
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]

	var residentOverhead int64
	for tIdx := range residentTensors {
		if !boundary.BoundaryInputs[tIdx] && !boundary.AllProduced[tIdx] {
			residentOverhead += FullTensorSize(p, tIdx)
		}
	}
	for tIdx := range boundary.BoundaryInputs {
		if residentTensors[tIdx] {
			residentOverhead += FullTensorSize(p, tIdx)
		}
	}

	availCap := p.FastMemoryCapacity - residentOverhead
	var results [][3]int

	if hasMatmul {
		numLHS, numRHS, numPW := 0, 0, 0
		numOut := len(boundary.BoundaryOutputs)

		for tIdx := range boundary.BoundaryInputs {
			if residentTensors[tIdx] {
				continue
			}
			role := InputTileRole(p, ops, tIdx)
			switch role {
			case "LHS":
				numLHS++
			case "RHS":
				numRHS++
			case "PW":
				numPW++
			}
		}

		for _, k := range []int{maxK, maxK / 2, maxK / 4} {
			if k <= 0 {
				continue
			}
			lo, hi := nw, outW
			for lo <= hi {
				mid := (lo + hi) / 2
				ws := int64(numLHS)*int64(k)*int64(nh) + int64(numRHS)*int64(mid)*int64(k) + int64(numPW+numOut)*int64(mid)*int64(nh)
				if ws <= availCap {
					lo = mid + 1
				} else {
					hi = mid - 1
				}
			}
			if hi >= nw {
				results = append(results, [3]int{MinInt(hi, outW), nh, k})
			}

			lo, hi = nh, outH
			for lo <= hi {
				mid := (lo + hi) / 2
				ws := int64(numLHS)*int64(k)*int64(mid) + int64(numRHS)*int64(nw)*int64(k) + int64(numPW+numOut)*int64(nw)*int64(mid)
				if ws <= availCap {
					lo = mid + 1
				} else {
					hi = mid - 1
				}
			}
			if hi >= nh {
				results = append(results, [3]int{nw, MinInt(hi, outH), k})
			}
		}
	} else {
		numIO := len(boundary.BoundaryInputs) + len(boundary.BoundaryOutputs)
		for tIdx := range boundary.BoundaryInputs {
			if residentTensors[tIdx] {
				numIO--
			}
		}
		if numIO > 0 {
			maxTileSize := availCap / int64(numIO)
			if maxTileSize > 0 {
				s := int(math.Sqrt(float64(maxTileSize)))
				s = (s / nw) * nw
				if s > 0 {
					results = append(results, [3]int{MinInt(s, outW), MinInt(s, outH), 1})
				}
			}
		}
	}

	return results
}

func findSmallestFeasible(p *Problem, ops []int, residentTensors map[int]bool) [3]int {
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]
	maxK := GetMaxK(p, ops)

	for w := nw; w >= 1; w /= 2 {
		for h := nh; h >= 1; h /= 2 {
			for k := maxK; k >= 1; k /= 2 {
				gran := [3]int{w, h, k}
				ws := ComputeWorkingSet(p, ops, gran, residentTensors)
				if ws <= p.FastMemoryCapacity {
					return gran
				}
				if !HasMatMul(p, ops) {
					break
				}
			}
		}
	}
	return [3]int{1, 1, 1}
}

func SnakeTraversal(nCols, nRows int) []int {
	order := make([]int, 0, nCols*nRows)
	for row := 0; row < nRows; row++ {
		if row%2 == 0 {
			for col := 0; col < nCols; col++ {
				order = append(order, row*nCols+col)
			}
		} else {
			for col := nCols - 1; col >= 0; col-- {
				order = append(order, row*nCols+col)
			}
		}
	}
	return order
}

func ColumnSnakeTraversal(nCols, nRows int) []int {
	order := make([]int, 0, nCols*nRows)
	for col := 0; col < nCols; col++ {
		if col%2 == 0 {
			for row := 0; row < nRows; row++ {
				order = append(order, row*nCols+col)
			}
		} else {
			for row := nRows - 1; row >= 0; row-- {
				order = append(order, row*nCols+col)
			}
		}
	}
	return order
}

func BestTraversal(p *Problem, ops []int, gran [3]int) []int {
	w, h, k := gran[0], gran[1], gran[2]
	primaryOutput := GetOutputTensor(p, ops)
	outT := p.Tensors[primaryOutput]
	nCols := CeilDiv(outT.Width, w)
	nRows := CeilDiv(outT.Height, h)

	if nCols*nRows <= 1 {
		return nil
	}

	if !HasMatMul(p, ops) {
		if nCols >= nRows {
			return SnakeTraversal(nCols, nRows)
		}
		return ColumnSnakeTraversal(nCols, nRows)
	}

	boundary := GetSubgraphBoundary(p, ops)

	var lhsBandwidth, rhsBandwidth int64
	for tIdx := range boundary.BoundaryInputs {
		role := InputTileRole(p, ops, tIdx)
		tileSize := InputTileSize(p, ops, tIdx, w, h, k)
		switch role {
		case "LHS":
			lhsBandwidth += tileSize
		case "RHS":
			rhsBandwidth += tileSize
		}
	}

	rowMajorSavings := lhsBandwidth * int64(nCols-1) * int64(nRows)
	colMajorSavings := rhsBandwidth * int64(nRows-1) * int64(nCols)

	if colMajorSavings > rowMajorSavings {
		return ColumnSnakeTraversal(nCols, nRows)
	}
	return SnakeTraversal(nCols, nRows)
}
