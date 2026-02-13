package main

import (
	"fmt"
	"sort"
)

// ScheduleEntry represents one subgraph in the execution order
type ScheduleEntry struct {
	Ops         []int
	Granularity [3]int
	Traversal   []int
	Retain      []int
	Latency     float64
}

// BuildSchedule is the main scheduling function
func BuildSchedule(p *Problem, gi *GraphInfo, groups [][]int) []ScheduleEntry {
	numGroups := len(groups)

	// Build group-level dependency graph
	groupOf := make(map[int]int)
	for gIdx, group := range groups {
		for _, opIdx := range group {
			groupOf[opIdx] = gIdx
		}
	}

	groupDeps := make([]map[int]bool, numGroups)
	groupDependents := make([]map[int]bool, numGroups)
	for i := range groups {
		groupDeps[i] = make(map[int]bool)
		groupDependents[i] = make(map[int]bool)
	}

	for gIdx, group := range groups {
		for _, opIdx := range group {
			for _, depOp := range gi.Dependencies[opIdx] {
				depGroup := groupOf[depOp]
				if depGroup != gIdx {
					groupDeps[gIdx][depGroup] = true
					groupDependents[depGroup][gIdx] = true
				}
			}
		}
	}

	groupBoundaryInputs := make([]map[int]bool, numGroups)
	for i, group := range groups {
		boundary := GetSubgraphBoundary(p, group)
		groupBoundaryInputs[i] = boundary.BoundaryInputs
	}

	inDegree := make([]int, numGroups)
	for i := range groups {
		inDegree[i] = len(groupDeps[i])
	}

	var schedule []int
	lastScheduled := -1

	remaining := make(map[int]bool)
	for i := range groups {
		remaining[i] = true
	}

	for len(schedule) < numGroups {
		var ready []int
		for gIdx := range remaining {
			if inDegree[gIdx] == 0 {
				ready = append(ready, gIdx)
			}
		}

		if len(ready) == 0 {
			fmt.Println("WARNING: cycle detected in group dependencies")
			for gIdx := range remaining {
				ready = append(ready, gIdx)
			}
		}

		if lastScheduled >= 0 && len(ready) > 1 {
			lastOutputs := GetSubgraphBoundary(p, groups[lastScheduled]).BoundaryOutputs
			lastInputs := groupBoundaryInputs[lastScheduled]

			sort.Slice(ready, func(i, j int) bool {
				scoreI := computeAffinity(p, groupBoundaryInputs[ready[i]], lastOutputs, lastInputs)
				scoreJ := computeAffinity(p, groupBoundaryInputs[ready[j]], lastOutputs, lastInputs)
				return scoreI > scoreJ
			})
		}

		chosen := ready[0]
		schedule = append(schedule, chosen)
		delete(remaining, chosen)
		lastScheduled = chosen

		for dep := range groupDependents[chosen] {
			inDegree[dep]--
		}
	}

	entries := make([]ScheduleEntry, numGroups)
	for i, gIdx := range schedule {
		ops := groups[gIdx]
		entries[i] = ScheduleEntry{
			Ops: ops,
		}
	}

	return entries
}

func computeAffinity(p *Problem, nextInputs map[int]bool, lastOutputs map[int]bool, lastInputs map[int]bool) float64 {
	score := 0.0
	for tIdx := range nextInputs {
		if lastOutputs[tIdx] {
			score += float64(FullTensorSize(p, tIdx)) * 2.0
		}
		if lastInputs[tIdx] {
			score += float64(FullTensorSize(p, tIdx)) * 1.0
		}
	}
	return score
}

// OptimizeSchedule takes initial groups and produces a fully optimized schedule
func OptimizeSchedule(p *Problem, gi *GraphInfo) *Solution {
	// Phase 1: Form initial groups via chain fusion
	chains := FindLinearChains(p, gi)
	fmt.Printf("  Found %d linear chains\n", len(chains))

	var allGroups [][]int
	for _, chain := range chains {
		if len(chain) <= 3 {
			groups := FuseChainGreedy(p, chain, make(map[int]bool))
			allGroups = append(allGroups, groups...)
		} else {
			groups := FuseChainDP(p, chain, make(map[int]bool))
			allGroups = append(allGroups, groups...)
		}
	}
	fmt.Printf("  Formed %d groups after chain fusion\n", len(allGroups))

	// Phase 2: Try cross-chain fusion for groups sharing large inputs
	allGroups = tryCrossChainFusion(p, gi, allGroups)
	fmt.Printf("  %d groups after cross-chain fusion\n", len(allGroups))

	// Phase 3: Order groups
	schedule := BuildSchedule(p, gi, allGroups)
	fmt.Printf("  Ordered %d schedule entries\n", len(schedule))

	// Phase 4: Optimize granularity
	for i := range schedule {
		resident := make(map[int]bool)
		if i > 0 {
			for _, tIdx := range schedule[i-1].Retain {
				resident[tIdx] = true
			}
		}

		gran := FindBestGranularity(p, schedule[i].Ops, resident)
		trav := BestTraversal(p, schedule[i].Ops, gran)
		schedule[i].Granularity = gran
		schedule[i].Traversal = trav
	}

	// Phase 5: Plan retention
	for i := range schedule {
		resident := make(map[int]bool)
		if i > 0 {
			for _, tIdx := range schedule[i-1].Retain {
				resident[tIdx] = true
			}
		}

		retain := PlanRetentionGlobal(p, i, schedule, resident)
		schedule[i].Retain = retain
	}

	// Phase 6: Re-optimize granularity
	for i := range schedule {
		resident := make(map[int]bool)
		if i > 0 {
			for _, tIdx := range schedule[i-1].Retain {
				resident[tIdx] = true
			}
		}

		retainAfter := schedule[i].Retain
		ws := ComputeWorkingSetWithRetained(p, schedule[i].Ops, schedule[i].Granularity, resident, retainAfter)
		if ws > p.FastMemoryCapacity {
			gran := FindBestGranularityWithRetain(p, schedule[i].Ops, resident, retainAfter)
			trav := BestTraversal(p, schedule[i].Ops, gran)
			schedule[i].Granularity = gran
			schedule[i].Traversal = trav
		}

		lat, err := EvaluateSubgraphDetailed(
			p, schedule[i].Ops, schedule[i].Granularity,
			schedule[i].Retain, schedule[i].Traversal, resident,
		)
		if err != nil {
			lat = 0
		}
		schedule[i].Latency = lat
	}

	// Phase 7: Prune
	schedule = pruneRetentions(p, schedule)

	subgraphs := make([]Subgraph, len(schedule))
	for i, entry := range schedule {
		subgraphs[i] = Subgraph{
			Ops:             entry.Ops,
			Granularity:     entry.Granularity,
			TensorsToRetain: entry.Retain,
			TraversalOrder:  entry.Traversal,
			SubgraphLatency: entry.Latency,
		}
	}

	return &Solution{Subgraphs: subgraphs}
}

func pruneRetentions(p *Problem, schedule []ScheduleEntry) []ScheduleEntry {
	improved := true
	for improved {
		improved = false
		for i := range schedule {
			if len(schedule[i].Retain) == 0 {
				continue
			}

			for rIdx := len(schedule[i].Retain) - 1; rIdx >= 0; rIdx-- {
				currentTotal := schedule[i].Latency
				if i+1 < len(schedule) {
					currentTotal += schedule[i+1].Latency
				}

				newRetain := make([]int, 0, len(schedule[i].Retain)-1)
				for j, t := range schedule[i].Retain {
					if j != rIdx {
						newRetain = append(newRetain, t)
					}
				}

				residentI := make(map[int]bool)
				if i > 0 {
					for _, t := range schedule[i-1].Retain {
						residentI[t] = true
					}
				}

				latI, errI := EvaluateSubgraphDetailed(
					p, schedule[i].Ops, schedule[i].Granularity,
					newRetain, schedule[i].Traversal, residentI,
				)
				if errI != nil {
					continue
				}

				newTotal := latI
				if i+1 < len(schedule) {
					residentNext := make(map[int]bool)
					for _, t := range newRetain {
						residentNext[t] = true
					}

					latNext, errNext := EvaluateSubgraphDetailed(
						p, schedule[i+1].Ops, schedule[i+1].Granularity,
						schedule[i+1].Retain, schedule[i+1].Traversal, residentNext,
					)
					if errNext != nil {
						continue
					}
					newTotal += latNext
				}

				if newTotal < currentTotal {
					schedule[i].Retain = newRetain
					schedule[i].Latency = latI
					if i+1 < len(schedule) {
						residentNext := make(map[int]bool)
						for _, t := range newRetain {
							residentNext[t] = true
						}
						latNext, _ := EvaluateSubgraphDetailed(
							p, schedule[i+1].Ops, schedule[i+1].Granularity,
							schedule[i+1].Retain, schedule[i+1].Traversal, residentNext,
						)
						schedule[i+1].Latency = latNext
					}
					improved = true
				}
			}
		}
	}

	return schedule
}
