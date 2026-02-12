package main

import "fmt"

// EvaluateSubgraph computes the latency of a single subgraph.
//
// The model works like this:
//   - The subgraph has a set of "boundary" tensors: inputs that come from
//     slow memory, and outputs that go to slow memory (or stay retained).
//   - "Ephemeral" tensors are intermediates within the subgraph — free.
//   - The output tensor determines the spatial tile grid.
//   - For each spatial tile × each k-step, we compute one "step".
//   - Each step: load input tile slices, compute, store output tile slice.
//   - Latency per step = max(compute_time, memory_time).
//
// For the baseline: raster order, no reuse between tiles.
func EvaluateSubgraph(
	p *Problem,
	sg *Subgraph,
	residentTensors map[int]bool, // tensors already in fast memory from previous subgraph
) (float64, error) {

	if len(sg.Ops) == 0 {
		return 0, fmt.Errorf("subgraph has no ops")
	}

	w, h, k := sg.Granularity[0], sg.Granularity[1], sg.Granularity[2]

	// Identify boundary tensors:
	// - "ephemeral" = produced AND consumed entirely within this subgraph
	// - boundary inputs = consumed but not produced within subgraph (must load from slow/fast mem)
	// - boundary outputs = produced but not consumed within subgraph (must store)

	producedInSubgraph := make(map[int]bool)
	consumedInSubgraph := make(map[int]bool)
	opsInSubgraph := make(map[int]bool)

	for _, opIdx := range sg.Ops {
		opsInSubgraph[opIdx] = true
		for _, t := range p.Ops[opIdx].Outputs {
			producedInSubgraph[t] = true
		}
		for _, t := range p.Ops[opIdx].Inputs {
			consumedInSubgraph[t] = true
		}
	}

	// Ephemeral: produced AND consumed within the subgraph
	ephemeral := make(map[int]bool)
	for t := range producedInSubgraph {
		if consumedInSubgraph[t] {
			ephemeral[t] = true
		}
	}

	// Boundary inputs: consumed but NOT ephemeral and NOT produced-only within subgraph
	// (i.e., they must come from outside)
	boundaryInputs := make(map[int]bool)
	for t := range consumedInSubgraph {
		if !producedInSubgraph[t] {
			boundaryInputs[t] = true
		}
	}

	// Boundary outputs: produced but NOT ephemeral
	boundaryOutputs := make(map[int]bool)
	for t := range producedInSubgraph {
		if !ephemeral[t] {
			boundaryOutputs[t] = true
		}
	}

	// Find the "primary" output tensor to determine the spatial grid.
	// For the baseline, we use the last op's output.
	lastOp := p.Ops[sg.Ops[len(sg.Ops)-1]]
	primaryOutput := lastOp.Outputs[0]
	outTensor := p.Tensors[primaryOutput]

	// Spatial tile counts
	nCols := CeilDiv(outTensor.Width, w)
	nRows := CeilDiv(outTensor.Height, h)
	nSpatial := nCols * nRows

	// K-steps: for MatMul ops, we need to iterate over the reduction dimension.
	// The reduction dimension K is determined by the inner dimension of the MatMul.
	// For a chain, we use the max K across all MatMuls in the subgraph.
	nK := 1
	for _, opIdx := range sg.Ops {
		op := p.Ops[opIdx]
		if op.OpType == "MatMul" {
			// For MatMul: LHS is inputs[0], RHS is inputs[1]
			// LHS shape: (Height, K) where K = LHS.Width
			// RHS shape: (K, Width) where K = RHS.Height
			lhs := p.Tensors[op.Inputs[0]]
			K := lhs.Width // inner/reduction dimension
			opK := CeilDiv(K, k)
			if opK > nK {
				nK = opK
			}
		}
	}

	// Compute cost per step:
	// Sum of base_costs for all ops. But if granularity < native, we still pay
	// full native cost (padding penalty — we produce less useful output per step).
	nativeW, nativeH := p.NativeGranularity[0], p.NativeGranularity[1]
	var computePerStep int64
	for _, opIdx := range sg.Ops {
		computePerStep += p.Ops[opIdx].BaseCost
	}

	// Padding: if tile is smaller than native, we still pay full cost but
	// the spatial grid is based on our chosen granularity, not native.
	// Actually, per the problem: "you pay the full compute cost of the native
	// size but produce less useful output". This means each step's compute
	// cost remains base_cost regardless of tile size, but the number of steps
	// increases because we produce smaller tiles.
	// The base_cost is the cost at native granularity. So compute per step
	// is always sum(base_costs), but the number of spatial tiles is based on
	// our chosen (w, h), not the native size.
	_ = nativeW
	_ = nativeH
	// Actually we need to be more careful. The base_cost is for one native-sized
	// tile. If our tile is larger than native, we'd need multiple native executions.
	// But the problem says granularity is typically ≤ native or equal.
	// For safety, compute cost should scale with how many native tiles fit in our tile.
	// Per the examples: at native 128×128, a 128×128 tile costs base_cost once,
	// a 64×64 tile also costs base_cost once (padding), and the total is
	// more steps × same per-step cost.

	// Memory cost per step (baseline: no reuse between tiles):
	// Load all boundary input tile slices from slow memory,
	// Store all boundary output tile slices to slow memory.
	// Memory time = total_bytes_transferred / bandwidth

	bw := float64(p.SlowMemoryBandwidth)

	totalLatency := 0.0

	// For baseline with raster order and no reuse:
	// Every tile loads everything fresh.
	for step := 0; step < nSpatial; step++ {
		for kStep := 0; kStep < nK; kStep++ {
			// Memory transfer for this step
			var memoryBytes int64

			// For each boundary input tensor, compute the tile size loaded
			for t := range boundaryInputs {
				if residentTensors[t] {
					continue // already in fast memory, no transfer needed
				}
				tileSize := inputTileSize(p, sg, t, w, h, k)
				memoryBytes += tileSize
			}

			// For boundary outputs: only on the LAST k-step do we evict
			if kStep == nK-1 {
				for t := range boundaryOutputs {
					retaining := false
					for _, rt := range sg.TensorsToRetain {
						if rt == t {
							retaining = true
							break
						}
					}
					if !retaining {
						// Evict to slow memory
						outTileSize := int64(w) * int64(h)
						memoryBytes += outTileSize
					}
				}
			}

			memTime := float64(memoryBytes) / bw
			compTime := float64(computePerStep)

			stepLatency := MaxFloat(compTime, memTime)
			totalLatency += stepLatency
		}
	}

	return totalLatency, nil
}

// inputTileSize computes the size of the tile slice for a given input tensor
// based on the subgraph's granularity and what role this tensor plays.
//
// For MatMul:
//   - LHS (inputs[0]): tile is k × h (k columns, h rows)
//   - RHS (inputs[1]): tile is w × k (w columns, k rows)
//
// For Pointwise:
//   - All inputs: tile is w × h
func inputTileSize(p *Problem, sg *Subgraph, tensorIdx int, w, h, k int) int64 {
	// Find which op consumes this tensor and in what position
	for _, opIdx := range sg.Ops {
		op := p.Ops[opIdx]
		for pos, inp := range op.Inputs {
			if inp == tensorIdx {
				if op.OpType == "MatMul" {
					if pos == 0 {
						// LHS: height=h, width=k
						return int64(k) * int64(h)
					}
					// RHS: height=k, width=w
					return int64(w) * int64(k)
				}
				// Pointwise
				return int64(w) * int64(h)
			}
		}
	}
	// Shouldn't reach here for a valid boundary input
	return int64(w) * int64(h)
}

// ComputeWorkingSet returns the peak fast memory needed for one execution step.
// This is the sum of all input tile slices + output tile slices that must be
// simultaneously resident.
func ComputeWorkingSet(p *Problem, sg *Subgraph, residentTensors map[int]bool) int64 {
	w, h, k := sg.Granularity[0], sg.Granularity[1], sg.Granularity[2]

	producedInSubgraph := make(map[int]bool)
	consumedInSubgraph := make(map[int]bool)
	for _, opIdx := range sg.Ops {
		for _, t := range p.Ops[opIdx].Outputs {
			producedInSubgraph[t] = true
		}
		for _, t := range p.Ops[opIdx].Inputs {
			consumedInSubgraph[t] = true
		}
	}

	ephemeral := make(map[int]bool)
	for t := range producedInSubgraph {
		if consumedInSubgraph[t] {
			ephemeral[t] = true
		}
	}

	var ws int64

	// Input tiles
	for t := range consumedInSubgraph {
		if !producedInSubgraph[t] || !ephemeral[t] {
			// This is a boundary input — must be in fast memory during compute
			if !producedInSubgraph[t] {
				ws += inputTileSize(p, sg, t, w, h, k)
			}
		}
	}

	// Output tiles (non-ephemeral)
	for t := range producedInSubgraph {
		if !ephemeral[t] {
			ws += int64(w) * int64(h)
		}
	}

	// Retained tensors from previous subgraphs also consume space
	for t := range residentTensors {
		if !consumedInSubgraph[t] && !producedInSubgraph[t] {
			// This tensor is sitting in fast memory but not used by this subgraph
			ws += int64(p.Tensors[t].Width) * int64(p.Tensors[t].Height)
		}
	}

	return ws
}

// EvaluateSolution computes total latency and validates the solution.
func EvaluateSolution(p *Problem, sol *Solution) (float64, error) {
	// Check all ops are covered
	coveredOps := make(map[int]bool)
	for _, sg := range sol.Subgraphs {
		for _, opIdx := range sg.Ops {
			coveredOps[opIdx] = true
		}
	}
	for i := range p.Ops {
		if !coveredOps[i] {
			return 0, fmt.Errorf("op %d not covered by any subgraph", i)
		}
	}

	totalLatency := 0.0
	resident := make(map[int]bool) // tensors currently in fast memory

	for i, sg := range sol.Subgraphs {
		// Check working set fits
		ws := ComputeWorkingSet(p, &sg, resident)
		if ws > p.FastMemoryCapacity {
			return 0, fmt.Errorf("subgraph %d: working set %d exceeds capacity %d",
				i, ws, p.FastMemoryCapacity)
		}

		lat, err := EvaluateSubgraph(p, &sg, resident)
		if err != nil {
			return 0, fmt.Errorf("subgraph %d: %w", i, err)
		}

		totalLatency += lat

		// Update residency: clear old, add retained
		resident = make(map[int]bool)
		for _, t := range sg.TensorsToRetain {
			resident[t] = true
		}
	}

	return totalLatency, nil
}

// Quick debug helper
func PrintSolutionSummary(p *Problem, sol *Solution) {
	total := 0.0
	for i, sg := range sol.Subgraphs {
		fmt.Printf("Subgraph %d: ops=%v gran=[%d,%d,%d] retain=%v latency=%.1f\n",
			i, sg.Ops, sg.Granularity[0], sg.Granularity[1], sg.Granularity[2],
			sg.TensorsToRetain, sg.SubgraphLatency)
		total += sg.SubgraphLatency
	}
	fmt.Printf("Total latency: %.1f\n", total)
}
