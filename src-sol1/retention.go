package main

import (
	"sort"
)

// RetentionCandidate represents a tensor we might retain
type RetentionCandidate struct {
	TensorIdx int
	Size      int64
	Savings   float64
	Score     float64 // savings / size
}

// PlanRetention decides which tensors to retain between subgraphs
func PlanRetention(
	p *Problem,
	currentSubgraph []int,
	nextSubgraph []int,
	gran [3]int,
	currentResidents map[int]bool,
) []int {

	if nextSubgraph == nil {
		return []int{}
	}

	// Identify tensors produced by current that are consumed by next
	currentBoundary := GetSubgraphBoundary(p, nil, currentSubgraph)
	nextBoundary := GetSubgraphBoundary(p, nil, nextSubgraph)

	candidates := []RetentionCandidate{}

	for t := range currentBoundary.BoundaryOutputs {
		if nextBoundary.BoundaryInputs[t] {
			tensor := p.Tensors[t]
			size := int64(tensor.Width) * int64(tensor.Height)

			// Estimate savings: avoiding reload
			savings := float64(size) / float64(p.SlowMemoryBandwidth)

			// For tiled access, the savings is per-tile, times number of tiles
			outT := p.Tensors[GetOutputTensor(p, nextSubgraph)]
			nCols := CeilDiv(outT.Width, gran[0])
			nRows := CeilDiv(outT.Height, gran[1])
			nSpatial := nCols * nRows

			savings *= float64(nSpatial)

			score := savings / float64(size)

			candidates = append(candidates, RetentionCandidate{
				TensorIdx: t,
				Size:      size,
				Savings:   savings,
				Score:     score,
			})
		}
	}

	// Also consider tensors already resident that are used by next
	for t := range currentResidents {
		if nextBoundary.BoundaryInputs[t] {
			tensor := p.Tensors[t]
			size := int64(tensor.Width) * int64(tensor.Height)

			outT := p.Tensors[GetOutputTensor(p, nextSubgraph)]
			nCols := CeilDiv(outT.Width, gran[0])
			nRows := CeilDiv(outT.Height, gran[1])
			nSpatial := nCols * nRows

			savings := (float64(size) / float64(p.SlowMemoryBandwidth)) * float64(nSpatial)
			score := savings / float64(size)

			candidates = append(candidates, RetentionCandidate{
				TensorIdx: t,
				Size:      size,
				Savings:   savings,
				Score:     score,
			})
		}
	}

	// Sort by score descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	// Greedily pack until we hit capacity
	nextWS := ComputeWorkingSet(p, nextSubgraph, gran, currentResidents)
	availableCapacity := p.FastMemoryCapacity - nextWS

	retained := []int{}
	usedCapacity := int64(0)

	for _, cand := range candidates {
		if usedCapacity+cand.Size <= availableCapacity {
			retained = append(retained, cand.TensorIdx)
			usedCapacity += cand.Size
		}
	}

	return retained
}
