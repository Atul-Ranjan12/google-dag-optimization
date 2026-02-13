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
						// LHS: [h, k]
						// If h > tensorHeight (padding), we still pay for h
						// But here h is the tile height.
						return int64(k) * int64(h)
					}
					// RHS: [k, w]
					return int64(w) * int64(k)
				}
				// Pointwise: [w, h]
				return int64(w) * int64(h)
			}
		}
	}
	return int64(w) * int64(h)
}

// InputTileRole returns "LHS", "RHS", or "PW"
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

// FullTensorSize returns size of entire tensor
func FullTensorSize(p *Problem, tIdx int) int64 {
	t := p.Tensors[tIdx]
	return int64(t.Width) * int64(t.Height)
}

// ComputeWorkingSet returns peak fast memory for one step
func ComputeWorkingSet(p *Problem, ops []int, gran [3]int, residentTensors map[int]bool) int64 {
	w, h, k := gran[0], gran[1], gran[2]
	boundary := GetSubgraphBoundary(p, ops)

	var ws int64

	for tIdx := range boundary.BoundaryInputs {
		if residentTensors[tIdx] {
			// Resident tensor occupies its FULL size, not just a tile
			ws += FullTensorSize(p, tIdx)
		} else {
			ws += InputTileSize(p, ops, tIdx, w, h, k)
		}
	}

	for range boundary.BoundaryOutputs {
		ws += int64(w) * int64(h)
	}

	// Retained tensors not used by this subgraph
	for tIdx := range residentTensors {
		if !boundary.BoundaryInputs[tIdx] && !boundary.AllProduced[tIdx] {
			ws += FullTensorSize(p, tIdx)
		}
	}

	return ws
}

// ComputeWorkingSetWithRetained computes working set including tensors we plan to retain
func ComputeWorkingSetWithRetained(p *Problem, ops []int, gran [3]int, residentTensors map[int]bool, retainedAfter []int) int64 {
	ws := ComputeWorkingSet(p, ops, gran, residentTensors)

	// Add retained output tensors that need to stay as full tensors
	boundary := GetSubgraphBoundary(p, ops)
	for _, tIdx := range retainedAfter {
		if boundary.BoundaryOutputs[tIdx] {
			// The output tile is w*h but we need full tensor for retention
			// We already counted w*h for the output; add the rest
			fullSize := FullTensorSize(p, tIdx)
			tileSize := int64(gran[0]) * int64(gran[1])
			if fullSize > tileSize {
				ws += fullSize - tileSize
			}
		}
	}

	return ws
}

// GetMaxK returns max reduction dimension
func GetMaxK(p *Problem, ops []int) int {
	maxK := 1
	for _, opIdx := range ops {
		op := p.Ops[opIdx]
		if op.OpType == "MatMul" {
			lhs := p.Tensors[op.Inputs[0]]
			if lhs.Width > maxK {
				maxK = lhs.Width
			}
		}
	}
	return maxK
}

// GetOutputTensor returns primary output tensor
func GetOutputTensor(p *Problem, ops []int) int {
	lastOp := p.Ops[ops[len(ops)-1]]
	return lastOp.Outputs[0]
}

// HasMatMul checks if any op in ops is a MatMul
func HasMatMul(p *Problem, ops []int) bool {
	for _, opIdx := range ops {
		if p.Ops[opIdx].OpType == "MatMul" {
			return true
		}
	}
	return false
}

type tileInputInfo struct {
	tensorIdx int
	role      string // "LHS", "RHS", "PW"
	tileSize  int64
	fullSize  int64
}

// EvaluateSubgraphDetailed computes latency with reuse model
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

	boundary := GetSubgraphBoundary(p, ops)

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

	if traversalOrder == nil || len(traversalOrder) != nSpatial {
		traversalOrder = make([]int, nSpatial)
		for i := 0; i < nSpatial; i++ {
			traversalOrder[i] = i
		}
	}

	var boundaryInputList []tileInputInfo
	for tIdx := range boundary.BoundaryInputs {
		role := InputTileRole(p, ops, tIdx)
		size := InputTileSize(p, ops, tIdx, w, h, k)
		full := FullTensorSize(p, tIdx)
		boundaryInputList = append(boundaryInputList, tileInputInfo{tIdx, role, size, full})
	}

	retainSet := make(map[int]bool)
	for _, tIdx := range tensorsToRetain {
		retainSet[tIdx] = true
	}

	totalLatency := 0.0
	prevRow := -1
	prevCol := -1

	for step := 0; step < nSpatial; step++ {
		tileIdx := traversalOrder[step]
		row := tileIdx / nCols
		col := tileIdx % nCols

		for kStep := 0; kStep < nK; kStep++ {
			var memoryBytes int64

			for _, info := range boundaryInputList {
				// Check if fully resident from previous subgraph
				if residentTensors[info.tensorIdx] {
					continue
				}

				canReuse := false
				if step > 0 && kStep == 0 {
					switch info.role {
					case "LHS":
						if row == prevRow {
							canReuse = true
						}
					case "RHS":
						if col == prevCol {
							canReuse = true
						}
					case "PW":
						// PW inputs change every spatial tile
						canReuse = false
					}
				} else if kStep > 0 {
					// Within k-steps: PW inputs don't change with k
					if info.role == "PW" {
						canReuse = true
					}
					// MatMul inputs (LHS[h,k], RHS[k,w]) change with k, so need reload
				}

				if !canReuse {
					memoryBytes += info.tileSize
				}
			}

			// Output eviction on last k-step
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

// QuickEstimate provides a fast latency estimate for search purposes
func QuickEstimate(
	p *Problem,
	ops []int,
	gran [3]int,
	residentTensors map[int]bool,
) float64 {
	if len(ops) == 0 {
		return math.Inf(1)
	}

	w, h, k := gran[0], gran[1], gran[2]
	if w <= 0 || h <= 0 || k <= 0 {
		return math.Inf(1)
	}

	boundary := GetSubgraphBoundary(p, ops)
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

	// Estimate memory with snake reuse
	var totalMemory float64

	for tIdx := range boundary.BoundaryInputs {
		if residentTensors[tIdx] {
			continue
		}
		role := InputTileRole(p, ops, tIdx)
		tileSize := float64(InputTileSize(p, ops, tIdx, w, h, k))

		switch role {
		case "LHS":
			// LHS reused across columns in same row
			totalMemory += tileSize * float64(nRows) * float64(nK)
		case "RHS":
			// RHS reused across rows in same column
			totalMemory += tileSize * float64(nCols) * float64(nK)
		case "PW":
			// PW loaded every spatial tile
			totalMemory += tileSize * float64(nSpatial)
		}
	}

	// Output eviction
	for range boundary.BoundaryOutputs {
		totalMemory += float64(int64(w)*int64(h)) * float64(nSpatial)
	}

	totalCompute := float64(computePerStep) * float64(nSpatial) * float64(nK)
	totalMemTime := totalMemory / bw

	return MaxFloat(totalCompute, totalMemTime)
}

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
			return 0, fmt.Errorf("subgraph %d: working set %d exceeds capacity %d", i, ws, p.FastMemoryCapacity)
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
