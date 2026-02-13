package main

import (
	"fmt"
	"math"
)

// InputTileSize computes the tile size for a boundary input tensor
func InputTileSize(p *Problem, ops []int, tensorIdx int, w, h, k int) int64 {
	for _, opIdx := range ops {
		op := p.Ops[opIdx]
		for pos, inp := range op.Inputs {
			if inp == tensorIdx {
				if op.OpType == "MatMul" {
					if pos == 0 {
						return int64(k) * int64(h) // LHS: h rows, k cols
					}
					return int64(w) * int64(k) // RHS: k rows, w cols
				}
				return int64(w) * int64(h) // Pointwise
			}
		}
	}
	return int64(w) * int64(h)
}

// InputTileRole returns "LHS", "RHS", or "PW" for how a tensor is used
func InputTileRole(p *Problem, ops []int, tensorIdx int) string {
	for _, opIdx := range ops {
		op := p.Ops[opIdx]
		for pos, inp := range op.Inputs {
			if inp == tensorIdx {
				if op.OpType == "MatMul" {
					if pos == 0 {
						return "LHS"
					}
					return "RHS"
				}
				return "PW"
			}
		}
	}
	return "PW"
}

// ComputeWorkingSet returns peak fast memory for one step
func ComputeWorkingSet(p *Problem, ops []int, gran [3]int, retainedFromPrev map[int]bool) int64 {
	w, h, k := gran[0], gran[1], gran[2]

	boundary := GetSubgraphBoundary(p, nil, ops)

	var ws int64

	// Input tiles for boundary inputs
	for tIdx := range boundary.BoundaryInputs {
		ws += InputTileSize(p, ops, tIdx, w, h, k)
	}

	// Output tiles for boundary outputs
	for range boundary.BoundaryOutputs {
		ws += int64(w) * int64(h)
	}

	// Retained tensors from previous subgraphs that are NOT used here
	for tIdx := range retainedFromPrev {
		if !boundary.BoundaryInputs[tIdx] && !boundary.AllProduced[tIdx] {
			ws += int64(p.Tensors[tIdx].Width) * int64(p.Tensors[tIdx].Height)
		}
	}

	return ws
}

// GetMaxK returns the maximum reduction dimension across MatMul ops in the subgraph
func GetMaxK(p *Problem, ops []int) int {
	maxK := 1
	for _, opIdx := range ops {
		op := p.Ops[opIdx]
		if op.OpType == "MatMul" {
			lhs := p.Tensors[op.Inputs[0]]
			K := lhs.Width
			if K > maxK {
				maxK = K
			}
		}
	}
	return maxK
}

// GetOutputTensor returns the primary output tensor for spatial grid computation
func GetOutputTensor(p *Problem, ops []int) int {
	lastOp := p.Ops[ops[len(ops)-1]]
	return lastOp.Outputs[0]
}

// SnakeTraversal generates a snake/zig-zag traversal order for nCols x nRows grid
func SnakeTraversal(nCols, nRows int) []int {
	order := make([]int, 0, nCols*nRows)
	for row := 0; row < nRows; row++ {
		if row%2 == 0 {
			// left to right
			for col := 0; col < nCols; col++ {
				order = append(order, row*nCols+col)
			}
		} else {
			// right to left
			for col := nCols - 1; col >= 0; col-- {
				order = append(order, row*nCols+col)
			}
		}
	}
	return order
}

// EvaluateSubgraphDetailed computes latency with intelligent reuse model
func EvaluateSubgraphDetailed(
	p *Problem,
	ops []int,
	gran [3]int,
	tensorsToRetain []int,
	traversalOrder []int,
	residentTensors map[int]bool,
) (float64, error) {

	if len(ops) == 0 {
		return 0, fmt.Errorf("empty ops")
	}

	w, h, k := gran[0], gran[1], gran[2]
	if w <= 0 || h <= 0 || k <= 0 {
		return 0, fmt.Errorf("invalid granularity [%d,%d,%d]", w, h, k)
	}

	boundary := GetSubgraphBoundary(p, nil, ops)

	primaryOutput := GetOutputTensor(p, ops)
	outT := p.Tensors[primaryOutput]

	nCols := CeilDiv(outT.Width, w)
	nRows := CeilDiv(outT.Height, h)
	nSpatial := nCols * nRows

	maxK := GetMaxK(p, ops)
	nK := CeilDiv(maxK, k)

	var computePerStep int64
	for _, opIdx := range ops {
		computePerStep += p.Ops[opIdx].BaseCost
	}

	bw := float64(p.SlowMemoryBandwidth)

	if traversalOrder == nil || len(traversalOrder) == 0 {
		traversalOrder = make([]int, nSpatial)
		for i := 0; i < nSpatial; i++ {
			traversalOrder[i] = i
		}
	}

	type inputInfo struct {
		tensorIdx int
		role      string
		tileSize  int64
	}

	var boundaryInputList []inputInfo
	for tIdx := range boundary.BoundaryInputs {
		role := InputTileRole(p, ops, tIdx)
		size := InputTileSize(p, ops, tIdx, w, h, k)
		boundaryInputList = append(boundaryInputList, inputInfo{tIdx, role, size})
	}

	retainSet := make(map[int]bool)
	for _, tIdx := range tensorsToRetain {
		retainSet[tIdx] = true
	}

	totalLatency := 0.0
	prevRow := -1
	prevCol := -1

	// Track what's currently resident for reuse
	currentlyResident := make(map[int]bool)
	for tIdx := range residentTensors {
		currentlyResident[tIdx] = true
	}

	for step := 0; step < nSpatial; step++ {
		tileIdx := traversalOrder[step]
		row := tileIdx / nCols
		col := tileIdx % nCols

		for kStep := 0; kStep < nK; kStep++ {
			var memoryBytes int64

			for _, info := range boundaryInputList {
				// Check if already resident from previous subgraph
				if residentTensors[info.tensorIdx] && step == 0 && kStep == 0 {
					// Already in fast memory, no load needed
					currentlyResident[info.tensorIdx] = true
					continue
				}

				// Check if we can reuse from previous tile in this subgraph
				canReuse := false
				if currentlyResident[info.tensorIdx] {
					if kStep == 0 && step > 0 {
						switch info.role {
						case "LHS":
							if row == prevRow {
								canReuse = true
							}
						case "RHS":
							if col == prevCol {
								canReuse = true
							}
						}
					} else if kStep > 0 && info.role == "PW" {
						// Pointwise inputs don't change across k-steps
						canReuse = true
					}
				}

				if !canReuse {
					memoryBytes += info.tileSize
					currentlyResident[info.tensorIdx] = true
				}
			}

			// Output: evict on last k-step, unless retained
			if kStep == nK-1 {
				for tIdx := range boundary.BoundaryOutputs {
					if !retainSet[tIdx] {
						memoryBytes += int64(w) * int64(h)
					}
				}
			}

			memTime := float64(memoryBytes) / bw
			compTime := float64(computePerStep)

			stepLatency := MaxFloat(compTime, memTime)
			totalLatency += stepLatency
		}

		prevRow = row
		prevCol = col
	}

	return totalLatency, nil
}

// EvaluateSubgraphSimple computes latency without detailed reuse (for quick estimates)
func EvaluateSubgraphSimple(
	p *Problem,
	ops []int,
	gran [3]int,
	tensorsToRetain []int,
	residentTensors map[int]bool,
) float64 {

	if len(ops) == 0 {
		return math.Inf(1)
	}

	w, h, k := gran[0], gran[1], gran[2]
	if w <= 0 || h <= 0 || k <= 0 {
		return math.Inf(1)
	}

	boundary := GetSubgraphBoundary(p, nil, ops)

	primaryOutput := GetOutputTensor(p, ops)
	outT := p.Tensors[primaryOutput]

	nCols := CeilDiv(outT.Width, w)
	nRows := CeilDiv(outT.Height, h)
	nSpatial := nCols * nRows

	maxK := GetMaxK(p, ops)
	nK := CeilDiv(maxK, k)

	var computePerStep int64
	for _, opIdx := range ops {
		computePerStep += p.Ops[opIdx].BaseCost
	}

	bw := float64(p.SlowMemoryBandwidth)

	retainSet := make(map[int]bool)
	if tensorsToRetain != nil {
		for _, tIdx := range tensorsToRetain {
			retainSet[tIdx] = true
		}
	}

	// Estimate with basic reuse for snake traversal
	hasMatMul := false
	for _, opIdx := range ops {
		if p.Ops[opIdx].OpType == "MatMul" {
			hasMatMul = true
			break
		}
	}

	var avgMemBytesPerStep int64
	if hasMatMul && nSpatial > 1 {
		// With snake traversal, we get some reuse
		// Estimate: LHS reused within rows, RHS reused at row transitions
		var lhsSize, rhsSize, pwSize int64
		for tIdx := range boundary.BoundaryInputs {
			if residentTensors[tIdx] {
				continue
			}
			role := InputTileRole(p, ops, tIdx)
			size := InputTileSize(p, ops, tIdx, w, h, k)
			switch role {
			case "LHS":
				lhsSize += size
			case "RHS":
				rhsSize += size
			case "PW":
				pwSize += size
			}
		}

		// Average over all tiles:
		// - LHS loaded nRows times (once per row)
		// - RHS loaded nCols times (once per col)
		// - PW loaded every tile
		totalLHSLoads := float64(lhsSize) * float64(nRows)
		totalRHSLoads := float64(rhsSize) * float64(nCols)
		totalPWLoads := float64(pwSize) * float64(nSpatial)

		avgMemBytesPerStep = int64((totalLHSLoads + totalRHSLoads + totalPWLoads) / float64(nSpatial))
	} else {
		// No reuse
		for tIdx := range boundary.BoundaryInputs {
			if residentTensors[tIdx] {
				continue
			}
			avgMemBytesPerStep += InputTileSize(p, ops, tIdx, w, h, k)
		}
	}

	var outBytesPerStep int64
	for range boundary.BoundaryOutputs {
		outBytesPerStep += int64(w) * int64(h)
	}

	totalLatency := 0.0
	for step := 0; step < nSpatial; step++ {
		for kStep := 0; kStep < nK; kStep++ {
			memBytes := avgMemBytesPerStep
			if kStep == nK-1 {
				memBytes += outBytesPerStep
			}
			memTime := float64(memBytes) / bw
			compTime := float64(computePerStep)
			totalLatency += MaxFloat(compTime, memTime)
		}
	}

	return totalLatency
}

// EvaluateSolution computes total latency and validates
func EvaluateSolution(p *Problem, sol *Solution) (float64, error) {
	coveredOps := make(map[int]bool)
	for _, sg := range sol.Subgraphs {
		for _, opIdx := range sg.Ops {
			coveredOps[opIdx] = true
		}
	}
	for i := range p.Ops {
		if !coveredOps[i] {
			return 0, fmt.Errorf("op %d not covered", i)
		}
	}

	totalLatency := 0.0
	resident := make(map[int]bool)

	for i, sg := range sol.Subgraphs {
		ws := ComputeWorkingSet(p, sg.Ops, sg.Granularity, resident)
		if ws > p.FastMemoryCapacity {
			return 0, fmt.Errorf("subgraph %d: working set %d exceeds capacity %d",
				i, ws, p.
					FastMemoryCapacity)
		}

		lat, err := EvaluateSubgraphDetailed(
			p, sg.Ops, sg.Granularity, sg.TensorsToRetain,
			sg.TraversalOrder, resident,
		)
		if err != nil {
			return 0, fmt.Errorf("subgraph %d: %w", i, err)
		}

		totalLatency += lat

		resident = make(map[int]bool)
		for _, tIdx := range sg.TensorsToRetain {
			resident[tIdx] = true
		}
	}

	return totalLatency, nil
}
