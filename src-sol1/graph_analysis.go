package main

// GraphInfo holds precomputed analysis of the DAG
type GraphInfo struct {
	// Which op produces each tensor (-1 if graph input)
	ProducerOf map[int]int
	// Which ops consume each tensor
	ConsumersOf map[int][]int
	// Graph inputs (tensors with no producer)
	GraphInputs map[int]bool
	// Graph outputs (tensors with no consumer)
	GraphOutputs map[int]bool
	// Topological order of ops
	TopoOrder []int
	// For each op, which ops must come before it
	Dependencies map[int][]int
	// For each op, which ops depend on it
	Dependents map[int][]int
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

	// Build producer/consumer maps
	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			gi.ProducerOf[t] = i
		}
		for _, t := range op.Inputs {
			gi.ConsumersOf[t] = append(gi.ConsumersOf[t], i)
		}
	}

	// Identify graph inputs and outputs
	for i := range p.Tensors {
		if _, produced := gi.ProducerOf[i]; !produced {
			gi.GraphInputs[i] = true
		}
		if len(gi.ConsumersOf[i]) == 0 {
			gi.GraphOutputs[i] = true
		}
	}

	// Build op-level dependency graph
	for i, op := range p.Ops {
		for _, t := range op.Inputs {
			if producer, exists := gi.ProducerOf[t]; exists {
				gi.Dependencies[i] = append(gi.Dependencies[i], producer)
				gi.Dependents[producer] = append(gi.Dependents[producer], i)
			}
		}
		// deduplicate
		gi.Dependencies[i] = uniqueInts(gi.Dependencies[i])
		gi.Dependents[i] = uniqueInts(gi.Dependents[i])
	}

	// Topological sort (Kahn's algorithm)
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

// FindLinearChains finds maximal chains where each op's output feeds exactly one consumer
func FindLinearChains(p *Problem, gi *GraphInfo) [][]int {
	visited := make(map[int]bool)
	var chains [][]int

	for _, opIdx := range gi.TopoOrder {
		if visited[opIdx] {
			continue
		}

		// Start a new chain from this op
		chain := []int{opIdx}
		visited[opIdx] = true

		// Extend forward
		current := opIdx
		for {
			// Check if current op has exactly one output tensor
			op := p.Ops[current]
			if len(op.Outputs) != 1 {
				break
			}
			outTensor := op.Outputs[0]

			// Check if that tensor has exactly one consumer
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

// GetSubgraphBoundary computes boundary inputs, outputs, and ephemeral tensors
type SubgraphBoundary struct {
	BoundaryInputs  map[int]bool // tensors that must be loaded from outside
	BoundaryOutputs map[int]bool // tensors produced that leave the subgraph
	Ephemeral       map[int]bool // tensors produced AND consumed within subgraph
	AllConsumed     map[int]bool
	AllProduced     map[int]bool
}

func GetSubgraphBoundary(p *Problem, gi *GraphInfo, ops []int) *SubgraphBoundary {
	sb := &SubgraphBoundary{
		BoundaryInputs:  make(map[int]bool),
		BoundaryOutputs: make(map[int]bool),
		Ephemeral:       make(map[int]bool),
		AllConsumed:     make(map[int]bool),
		AllProduced:     make(map[int]bool),
	}

	opsSet := make(map[int]bool)
	for _, op := range ops {
		opsSet[op] = true
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

	// Ephemeral: produced AND consumed within subgraph
	for t := range sb.AllProduced {
		if sb.AllConsumed[t] {
			sb.Ephemeral[t] = true
		}
	}

	// Boundary inputs: consumed but not produced within subgraph
	for t := range sb.AllConsumed {
		if !sb.AllProduced[t] {
			sb.BoundaryInputs[t] = true
		}
	}

	// Boundary outputs: produced but not ephemeral
	for t := range sb.AllProduced {
		if !sb.Ephemeral[t] {
			sb.BoundaryOutputs[t] = true
		}
	}

	return sb
}
