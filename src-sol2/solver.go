package main

import (
	"fmt"
)

// SolveOptimized is the main solver entry point
func SolveOptimized(p *Problem) *Solution {
	fmt.Println("  Running sol-2 optimized solver...")

	// Phase 1: Analyze graph
	gi := AnalyzeGraph(p)
	fmt.Printf("  Graph: %d ops, %d graph inputs, %d graph outputs\n",
		len(p.Ops), len(gi.GraphInputs), len(gi.GraphOutputs))

	// Phase 2-7: Full optimization pipeline
	sol := OptimizeSchedule(p, gi)

	// Final verification
	totalLat, err := EvaluateSolution(p, sol)
	if err != nil {
		fmt.Printf("  WARNING: Validation failed: %v\n", err)
		fmt.Println("  Attempting recovery...")
		sol = recoverSolution(p, gi, sol)
		totalLat, err = EvaluateSolution(p, sol)
		if err != nil {
			fmt.Printf("  FATAL: Recovery failed: %v\n", err)
			// Last resort: baseline
			fmt.Println("  Falling back to baseline...")
			sol = baselineSolution(p, gi)
			totalLat, _ = EvaluateSolution(p, sol)
		}
	}

	fmt.Printf("  Final latency: %.1f\n", totalLat)
	return sol
}

// recoverSolution attempts to fix a broken solution
func recoverSolution(p *Problem, gi *GraphInfo, broken *Solution) *Solution {
	// Strategy: keep the grouping but recompute everything else conservatively
	var subgraphs []Subgraph

	resident := make(map[int]bool)

	for i, brokenSG := range broken.Subgraphs {
		ops := brokenSG.Ops

		// Verify ops are topologically valid
		ops = sortOpsTopologically(gi, ops)

		// Find a granularity that fits with current residency
		gran := FindBestGranularity(p, ops, resident)

		// Check working set
		ws := ComputeWorkingSet(p, ops, gran, resident)
		if ws > p.FastMemoryCapacity {
			// Split the group into individual ops
			for _, opIdx := range ops {
				singleOps := []int{opIdx}
				singleGran := FindBestGranularity(p, singleOps, resident)
				singleWS := ComputeWorkingSet(p, singleOps, singleGran, resident)

				if singleWS > p.FastMemoryCapacity {
					// Need to evict retained tensors
					resident = make(map[int]bool)
					singleGran = FindBestGranularity(p, singleOps, resident)
				}

				trav := BestTraversal(p, singleOps, singleGran)
				lat, err := EvaluateSubgraphDetailed(p, singleOps, singleGran, nil, trav, resident)
				if err != nil {
					lat = 0
				}

				subgraphs = append(subgraphs, Subgraph{
					Ops:             singleOps,
					Granularity:     singleGran,
					TensorsToRetain: []int{},
					TraversalOrder:  trav,
					SubgraphLatency: lat,
				})

				resident = make(map[int]bool)
			}
			continue
		}

		trav := BestTraversal(p, ops, gran)

		// Try simple retention for next subgraph
		var retain []int
		if i+1 < len(broken.Subgraphs) {
			nextOps := sortOpsTopologically(gi, broken.Subgraphs[i+1].Ops)
			nextGran := FindBestGranularity(p, nextOps, make(map[int]bool))
			retain = PlanRetentionSimple(p, ops, nextOps, gran, nextGran, resident)

			// Verify retention fits
			wsRetain := ComputeWorkingSetWithRetained(p, ops, gran, resident, retain)
			if wsRetain > p.FastMemoryCapacity {
				retain = []int{} // drop all retention
			}
		}

		lat, err := EvaluateSubgraphDetailed(p, ops, gran, retain, trav, resident)
		if err != nil {
			lat = 0
		}

		subgraphs = append(subgraphs, Subgraph{
			Ops:             ops,
			Granularity:     gran,
			TensorsToRetain: retain,
			TraversalOrder:  trav,
			SubgraphLatency: lat,
		})

		// Update resident
		resident = make(map[int]bool)
		for _, tIdx := range retain {
			resident[tIdx] = true
		}
	}

	return &Solution{Subgraphs: subgraphs}
}

// baselineSolution produces a safe fallback: one op per subgraph, no retention
func baselineSolution(p *Problem, gi *GraphInfo) *Solution {
	var subgraphs []Subgraph

	for _, opIdx := range gi.TopoOrder {
		ops := []int{opIdx}
		gran := FindBestGranularity(p, ops, make(map[int]bool))
		trav := BestTraversal(p, ops, gran)

		lat, err := EvaluateSubgraphDetailed(p, ops, gran, nil, trav, make(map[int]bool))
		if err != nil {
			lat = 0
		}

		subgraphs = append(subgraphs, Subgraph{
			Ops:             ops,
			Granularity:     gran,
			TensorsToRetain: []int{},
			TraversalOrder:  trav,
			SubgraphLatency: lat,
		})
	}

	return &Solution{Subgraphs: subgraphs}
}

// PrintSolutionSummary prints a human-readable summary
func PrintSolutionSummary(p *Problem, sol *Solution) {
	total := 0.0
	for i, sg := range sol.Subgraphs {
		fmt.Printf("  SG %d: ops=%v gran=[%d,%d,%d] retain=%v lat=%.1f\n",
			i, sg.Ops, sg.Granularity[0], sg.Granularity[1], sg.Granularity[2],
			sg.TensorsToRetain, sg.SubgraphLatency)
		total += sg.SubgraphLatency
	}
	fmt.Printf("  Total: %.1f\n", total)
}
