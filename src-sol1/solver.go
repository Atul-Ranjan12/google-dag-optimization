package main

import (
	"fmt"
)

// SolveOptimized is the main optimized solver
func SolveOptimized(p *Problem) *Solution {
	fmt.Println("Running optimized solver...")

	// Phase 1: Analyze graph
	gi := AnalyzeGraph(p)
	fmt.Printf("Graph analysis: %d ops, %d inputs, %d outputs\n",
		len(p.Ops), len(gi.GraphInputs), len(gi.GraphOutputs))

	// Phase 2: Form subgraphs via fusion
	fmt.Println("Forming subgraphs via fusion...")
	candidates := FormSubgraphs(p, gi)
	fmt.Printf("Formed %d initial subgraph candidates\n", len(candidates))

	// Phase 3: Optimize granularity and traversal per subgraph
	fmt.Println("Optimizing granularities and traversal orders...")
	subgraphs := make([]Subgraph, len(candidates))
	residentTensors := make(map[int]bool)

	for i, cand := range candidates {
		gran, trav, lat := OptimizeSubgraphGranularity(p, cand.Ops, residentTensors)

		// Phase 4: Plan retention for next subgraph
		var nextCand []int
		if i+1 < len(candidates) {
			nextCand = candidates[i+1].Ops
		}

		retention := PlanRetention(p, cand.Ops, nextCand, gran, residentTensors)

		subgraphs[i] = Subgraph{
			Ops:             cand.Ops,
			Granularity:     gran,
			TensorsToRetain: retention,
			TraversalOrder:  trav,
			SubgraphLatency: lat,
		}

		// Update resident tensors for next iteration
		residentTensors = make(map[int]bool)
		for _, t := range retention {
			residentTensors[t] = true
		}

		fmt.Printf("  Subgraph %d: ops=%v gran=[%d,%d,%d] latency=%.1f retain=%v\n",
			i, cand.Ops, gran[0], gran[1], gran[2], lat, retention)
	}

	// Phase 5: Final verification and latency computation
	fmt.Println("Verifying solution...")
	sol := &Solution{Subgraphs: subgraphs}

	totalLat, err := EvaluateSolution(p, sol)
	if err != nil {
		fmt.Printf("WARNING: Solution validation error: %v\n", err)
		// Try to fix by recomputing latencies
		for i := range sol.Subgraphs {
			sg := &sol.Subgraphs[i]
			resident := make(map[int]bool)
			if i > 0 {
				for _, t := range sol.Subgraphs[i-1].TensorsToRetain {
					resident[t] = true
				}
			}
			lat, _ := EvaluateSubgraphDetailed(
				p, sg.Ops, sg.Granularity, sg.TensorsToRetain,
				sg.TraversalOrder, resident,
			)
			sg.SubgraphLatency = lat
		}
		totalLat, _ = EvaluateSolution(p, sol)
	}

	fmt.Printf("Total latency: %.1f\n", totalLat)

	return sol
}

// PrintSolutionSummary prints a human-readable summary
func PrintSolutionSummary(p *Problem, sol *Solution) {
	total := 0.0
	for i, sg := range sol.Subgraphs {
		fmt.Printf("Subgraph %d: ops=%v gran=[%d,%d,%d] retain=%v lat=%.1f\n",
			i, sg.Ops, sg.Granularity[0], sg.Granularity[1], sg.Granularity[2],
			sg.TensorsToRetain, sg.SubgraphLatency)
		total += sg.SubgraphLatency
	}
	fmt.Printf("Total latency: %.1f\n", total)
}
