package main

import "fmt"

// TopologicalSort returns op indices in dependency order.
// An op can only run after all ops that produce its inputs have run.
func TopologicalSort(p *Problem) []int {
	numOps := len(p.Ops)

	// Build: which op produces each tensor?
	producerOf := make(map[int]int) // tensor -> op index
	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			producerOf[t] = i
		}
	}

	// Build adjacency: op dependencies
	// op A depends on op B if B produces a tensor that A consumes
	inDegree := make([]int, numOps)
	dependents := make([][]int, numOps) // dependents[B] = list of ops that depend on B

	for i, op := range p.Ops {
		for _, t := range op.Inputs {
			if producer, exists := producerOf[t]; exists {
				dependents[producer] = append(dependents[producer], i)
				inDegree[i]++
			}
			// If no producer, it's a graph input tensor — no dependency
		}
	}

	// Kahn's algorithm
	queue := make([]int, 0)
	for i := 0; i < numOps; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}

	order := make([]int, 0, numOps)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		order = append(order, node)
		for _, dep := range dependents[node] {
			inDegree[dep]--
			if inDegree[dep] == 0 {
				queue = append(queue, dep)
			}
		}
	}

	return order
}

// FindGranularity finds the largest granularity [w, h, k] that fits in fast memory
// for a single-op subgraph. Starts at native size and halves dimensions until it fits.
func FindGranularity(p *Problem, opIdx int) [3]int {
	op := p.Ops[opIdx]
	nw, nh := p.NativeGranularity[0], p.NativeGranularity[1]

	// Determine k: for MatMul, k = full inner dimension initially, then shrink.
	// For Pointwise, k = 1.
	fullK := 1
	if op.OpType == "MatMul" {
		lhs := p.Tensors[op.Inputs[0]]
		fullK = lhs.Width // inner dimension
	}

	// Try granularities from large to small.
	// We try: native w/h first, then halve w, then halve h, then halve k, repeat.
	// Simple approach: try all power-of-2 fractions of native.
	for w := nw; w >= 1; w /= 2 {
		for h := nh; h >= 1; h /= 2 {
			for k := fullK; k >= 1; k /= 2 {
				// Compute working set for this granularity
				sg := &Subgraph{
					Ops:         []int{opIdx},
					Granularity: [3]int{w, h, k},
				}
				ws := ComputeWorkingSet(p, sg, nil)
				if ws <= p.FastMemoryCapacity {
					return [3]int{w, h, k}
				}
				if op.OpType != "MatMul" {
					break // k doesn't matter for Pointwise
				}
			}
		}
	}

	// Fallback: smallest possible
	return [3]int{1, 1, 1}
}

// SolveBaseline produces the simplest valid solution:
// one op per subgraph, topological order, no retention.
func SolveBaseline(p *Problem) *Solution {
	order := TopologicalSort(p)

	subgraphs := make([]Subgraph, 0, len(order))

	for _, opIdx := range order {
		gran := FindGranularity(p, opIdx)

		sg := Subgraph{
			Ops:             []int{opIdx},
			Granularity:     gran,
			TensorsToRetain: []int{},
			TraversalOrder:  nil,
		}

		// Compute latency
		lat, err := EvaluateSubgraph(p, &sg, nil)
		if err != nil {
			fmt.Printf("WARNING: failed to evaluate op %d: %v\n", opIdx, err)
			lat = 0
		}
		sg.SubgraphLatency = lat

		subgraphs = append(subgraphs, sg)
	}

	return &Solution{Subgraphs: subgraphs}
}
