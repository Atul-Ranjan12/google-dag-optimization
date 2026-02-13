package main

import "sort"

// GraphInfo holds precomputed analysis of the DAG
type GraphInfo struct {
	ProducerOf   map[int]int
	ConsumersOf  map[int][]int
	GraphInputs  map[int]bool
	GraphOutputs map[int]bool
	TopoOrder    []int
	Dependencies map[int][]int
	Dependents   map[int][]int
}

func AnalyzeGraph(p *Problem) *GraphInfo {
	gi := &GraphInfo{
		ProducerOf:   make(map[int]int),
		ConsumersOf:  make(map[int][]int),
		GraphInputs:  make(map[int]bool),
		GraphOutputs: make(map[int]bool),
		Dependencies: make(map[int][]int),
		Dependents:   make(map[int][]int),
	}

	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			gi.ProducerOf[t] = i
		}
		for _, t := range op.Inputs {
			gi.ConsumersOf[t] = append(gi.ConsumersOf[t], i)
		}
	}

	for i := range p.Tensors {
		if _, produced := gi.ProducerOf[i]; !produced {
			gi.GraphInputs[i] = true
		}
		if len(gi.ConsumersOf[i]) == 0 {
			gi.GraphOutputs[i] = true
		}
	}

	for i, op := range p.Ops {
		seen := make(map[int]bool)
		for _, t := range op.Inputs {
			if producer, exists := gi.ProducerOf[t]; exists {
				if !seen[producer] {
					gi.Dependencies[i] = append(gi.Dependencies[i], producer)
					gi.Dependents[producer] = append(gi.Dependents[producer], i)
					seen[producer] = true
				}
			}
		}
	}

	gi.TopoOrder = topologicalSort(p, gi)
	return gi
}

func topologicalSort(p *Problem, gi *GraphInfo) []int {
	numOps := len(p.Ops)
	inDegree := make([]int, numOps)
	for i := 0; i < numOps; i++ {
		inDegree[i] = len(gi.Dependencies[i])
	}

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
		for _, dep := range gi.Dependents[node] {
			inDegree[dep]--
			if inDegree[dep] == 0 {
				queue = append(queue, dep)
			}
		}
	}
	return order
}

// SubgraphBoundary computes boundary info for a set of ops
type SubgraphBoundary struct {
	BoundaryInputs  map[int]bool
	BoundaryOutputs map[int]bool
	Ephemeral       map[int]bool
	AllConsumed     map[int]bool
	AllProduced     map[int]bool
}

func GetSubgraphBoundary(p *Problem, ops []int) *SubgraphBoundary {
	sb := &SubgraphBoundary{
		BoundaryInputs:  make(map[int]bool),
		BoundaryOutputs: make(map[int]bool),
		Ephemeral:       make(map[int]bool),
		AllConsumed:     make(map[int]bool),
		AllProduced:     make(map[int]bool),
	}

	for _, opIdx := range ops {
		op := p.Ops[opIdx]
		for _, t := range op.Outputs {
			sb.AllProduced[t] = true
		}
		for _, t := range op.Inputs {
			sb.AllConsumed[t] = true
		}
	}

	for t := range sb.AllProduced {
		if sb.AllConsumed[t] {
			sb.Ephemeral[t] = true
		}
	}

	for t := range sb.AllConsumed {
		if !sb.AllProduced[t] {
			sb.BoundaryInputs[t] = true
		}
	}

	for t := range sb.AllProduced {
		if !sb.Ephemeral[t] {
			sb.BoundaryOutputs[t] = true
		}
	}

	return sb
}

// FindLinearChains finds maximal chains
func FindLinearChains(p *Problem, gi *GraphInfo) [][]int {
	visited := make(map[int]bool)
	var chains [][]int

	for _, opIdx := range gi.TopoOrder {
		if visited[opIdx] {
			continue
		}

		chain := []int{opIdx}
		visited[opIdx] = true

		current := opIdx
		for {
			op := p.Ops[current]
			if len(op.Outputs) != 1 {
				break
			}
			outTensor := op.Outputs[0]
			consumers := gi.ConsumersOf[outTensor]
			if len(consumers) != 1 {
				break
			}
			next := consumers[0]
			if visited[next] {
				break
			}
			chain = append(chain, next)
			visited[next] = true
			current = next
		}

		chains = append(chains, chain)
	}

	return chains
}

// AllAncestorOps returns all ops that must execute before opIdx (transitively)
func AllAncestorOps(gi *GraphInfo, opIdx int) map[int]bool {
	ancestors := make(map[int]bool)
	var dfs func(int)
	dfs = func(op int) {
		for _, dep := range gi.Dependencies[op] {
			if !ancestors[dep] {
				ancestors[dep] = true
				dfs(dep)
			}
		}
	}
	dfs(opIdx)
	return ancestors
}

// --- Shared Helpers Moved Here to Avoid Duplication ---

// canFuseGroups checks if two groups can be fused without creating cycles
func canFuseGroups(p *Problem, gi *GraphInfo, g1, g2 []int) bool {
	combined := append(append([]int{}, g1...), g2...)
	return isTopologicallyValid(p, gi, combined)
}

// isTopologicallyValid checks if a set of ops can form a valid subgraph
func isTopologicallyValid(p *Problem, gi *GraphInfo, ops []int) bool {
	opSet := make(map[int]bool)
	for _, op := range ops {
		opSet[op] = true
	}

	for _, op := range ops {
		for _, dep := range gi.Dependencies[op] {
			if !opSet[dep] {
				depAncestors := AllAncestorOps(gi, dep)
				for _, other := range ops {
					if other != op && depAncestors[other] {
						return false
					}
				}
			}
		}
	}

	return true
}

// sortOpsTopologically sorts ops according to the global topological order
func sortOpsTopologically(gi *GraphInfo, ops []int) []int {
	orderMap := make(map[int]int)
	for i, op := range gi.TopoOrder {
		orderMap[op] = i
	}

	sorted := make([]int, len(ops))
	copy(sorted, ops)
	sort.Slice(sorted, func(i, j int) bool {
		return orderMap[sorted[i]] < orderMap[sorted[j]]
	})
	return sorted
}
