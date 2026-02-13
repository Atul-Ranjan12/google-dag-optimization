package main

import (
	"sort"
)

// RetentionCandidate represents a tensor we might retain
type RetentionCandidate struct {
	TensorIdx int
	Size      int64
	Savings   float64 // estimated bandwidth savings
}

// PlanRetentionGlobal decides which tensors to retain, looking at all future subgraphs
func PlanRetentionGlobal(
	p *Problem,
	currentIdx int,
	schedule []ScheduleEntry,
	currentResident map[int]bool,
) []int {

	if currentIdx >= len(schedule)-1 {
		return []int{}
	}

	current := schedule[currentIdx]
	currentBoundary := GetSubgraphBoundary(p, current.Ops)

	// Collect all tensors that could be retained
	retainableTensors := make(map[int]bool)
	for tIdx := range currentBoundary.BoundaryOutputs {
		retainableTensors[tIdx] = true
	}
	for tIdx := range currentResident {
		retainableTensors[tIdx] = true
	}

	// For each retainable tensor, compute the savings from retaining it
	bw := float64(p.SlowMemoryBandwidth)
	var candidates []RetentionCandidate

	for tIdx := range retainableTensors {
		size := FullTensorSize(p, tIdx)

		savings := 0.0

		// Immediate next subgraph gets the biggest benefit
		nextIdx := currentIdx + 1
		if nextIdx < len(schedule) {
			nextBoundary := GetSubgraphBoundary(p, schedule[nextIdx].Ops)
			if nextBoundary.BoundaryInputs[tIdx] {
				nextGran := schedule[nextIdx].Granularity
				nextOutT := p.Tensors[GetOutputTensor(p, schedule[nextIdx].Ops)]

				nCols := CeilDiv(nextOutT.Width, nextGran[0])
				nRows := CeilDiv(nextOutT.Height, nextGran[1])
				nSpatial := nCols * nRows

				role := InputTileRole(p, schedule[nextIdx].Ops, tIdx)
				tileSize := InputTileSize(p, schedule[nextIdx].Ops, tIdx, nextGran[0], nextGran[1], nextGran[2])

				maxK := GetMaxK(p, schedule[nextIdx].Ops)
				nK := CeilDiv(maxK, nextGran[2])

				var loads int
				switch role {
				case "LHS":
					loads = nRows * nK
				case "RHS":
					loads = nCols * nK
				case "PW":
					loads = nSpatial
				}

				savings = float64(tileSize) * float64(loads) / bw

				// Also save on eviction from current subgraph
				if currentBoundary.BoundaryOutputs[tIdx] {
					savings += float64(size) / bw
				}
			}
		}

		// Look further ahead (with decay)
		for lookahead := 2; lookahead <= MinInt(4, len(schedule)-currentIdx-1); lookahead++ {
			futureIdx := currentIdx + lookahead
			futureBoundary := GetSubgraphBoundary(p, schedule[futureIdx].Ops)
			if futureBoundary.BoundaryInputs[tIdx] {
				savings += float64(size) / bw * 0.3
			}
		}

		if savings > 0 {
			candidates = append(candidates, RetentionCandidate{
				TensorIdx: tIdx,
				Size:      size,
				Savings:   savings,
			})
		}
	}

	// Sort by savings/size ratio (bang per buck)
	sort.Slice(candidates, func(i, j int) bool {
		ri := candidates[i].Savings / float64(candidates[i].Size)
		rj := candidates[j].Savings / float64(candidates[j].Size)
		return ri > rj
	})

	// Greedily pack: find available capacity
	nextOps := schedule[currentIdx+1].Ops
	nextGran := schedule[currentIdx+1].Granularity

	// Compute base working set of next subgraph with no retained tensors
	baseWS := ComputeWorkingSet(p, nextOps, nextGran, make(map[int]bool))
	availableCapacity := p.FastMemoryCapacity - baseWS

	// But we also need to account for resident tensors that won't be consumed by the next subgraph
	// If we retain tensor T and next subgraph doesn't use it, it still sits in fast memory
	nextBoundary := GetSubgraphBoundary(p, nextOps)

	var retained []int
	usedCapacity := int64(0)

	for _, cand := range candidates {
		tIdx := cand.TensorIdx

		// Compute the additional capacity cost of retaining this tensor
		additionalCost := cand.Size

		// If the next subgraph uses this tensor as a boundary input, then
		// ComputeWorkingSet already counted its tile size. Retaining it means
		// we replace the tile-sized entry with full-tensor-sized entry.
		if nextBoundary.BoundaryInputs[tIdx] {
			tileSize := InputTileSize(p, nextOps, tIdx, nextGran[0], nextGran[1], nextGran[2])
			additionalCost = cand.Size - tileSize
			// If full tensor is smaller than or equal to tile (small tensors), cost might be 0 or negative
			if additionalCost < 0 {
				additionalCost = 0
			}
		}

		if usedCapacity+additionalCost <= availableCapacity {
			retained = append(retained, tIdx)
			usedCapacity += additionalCost
		}
	}

	return retained
}

// PlanRetentionSimple is a simpler retention planner for when we don't have full schedule
func PlanRetentionSimple(
	p *Problem,
	currentOps []int,
	nextOps []int,
	currentGran [3]int,
	nextGran [3]int,
	currentResident map[int]bool,
) []int {

	if nextOps == nil {
		return []int{}
	}

	currentBoundary := GetSubgraphBoundary(p, currentOps)
	nextBoundary := GetSubgraphBoundary(p, nextOps)

	bw := float64(p.SlowMemoryBandwidth)

	type candidate struct {
		tIdx    int
		size    int64
		savings float64
	}

	var candidates []candidate

	// Check outputs of current subgraph
	for tIdx := range currentBoundary.BoundaryOutputs {
		if nextBoundary.BoundaryInputs[tIdx] {
			size := FullTensorSize(p, tIdx)

			nextOutT := p.Tensors[GetOutputTensor(p, nextOps)]
			nCols := CeilDiv(nextOutT.Width, nextGran[0])
			nRows := CeilDiv(nextOutT.Height, nextGran[1])
			nSpatial := nCols * nRows

			role := InputTileRole(p, nextOps, tIdx)
			tileSize := InputTileSize(p, nextOps, tIdx, nextGran[0], nextGran[1], nextGran[2])

			maxK := GetMaxK(p, nextOps)
			nK := CeilDiv(maxK, nextGran[2])

			var loads int
			switch role {
			case "LHS":
				loads = nRows * nK
			case "RHS":
				loads = nCols * nK
			case "PW":
				loads = nSpatial
			}

			savings := float64(tileSize)*float64(loads)/bw + float64(size)/bw
			candidates = append(candidates, candidate{tIdx, size, savings})
		}
	}

	// Check currently resident tensors
	for tIdx := range currentResident {
		if nextBoundary.BoundaryInputs[tIdx] {
			size := FullTensorSize(p, tIdx)

			nextOutT := p.Tensors[GetOutputTensor(p, nextOps)]
			nCols := CeilDiv(nextOutT.Width, nextGran[0])
			nRows := CeilDiv(nextOutT.Height, nextGran[1])
			nSpatial := nCols * nRows

			role := InputTileRole(p, nextOps, tIdx)
			tileSize := InputTileSize(p, nextOps, tIdx, nextGran[0], nextGran[1], nextGran[2])

			maxK := GetMaxK(p, nextOps)
			nK := CeilDiv(maxK, nextGran[2])

			var loads int
			switch role {
			case "LHS":
				loads = nRows * nK
			case "RHS":
				loads = nCols * nK
			case "PW":
				loads = nSpatial
			}

			savings := float64(tileSize) * float64(loads) / bw
			candidates = append(candidates, candidate{tIdx, size, savings})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		ri := candidates[i].savings / float64(candidates[i].size)
		rj := candidates[j].savings / float64(candidates[j].size)
		return ri > rj
	})

	baseWS := ComputeWorkingSet(p, nextOps, nextGran, make(map[int]bool))
	availableCapacity := p.FastMemoryCapacity - baseWS

	var retained []int
	usedCapacity := int64(0)

	for _, cand := range candidates {
		additionalCost := cand.size
		if nextBoundary.BoundaryInputs[cand.tIdx] {
			tileSize := InputTileSize(p, nextOps, cand.tIdx, nextGran[0], nextGran[1], nextGran[2])
			additionalCost = cand.size - tileSize
			if additionalCost < 0 {
				additionalCost = 0
			}
		}

		if usedCapacity+additionalCost <= availableCapacity {
			retained = append(retained, cand.tIdx)
			usedCapacity += additionalCost
		}
	}

	return retained
}
