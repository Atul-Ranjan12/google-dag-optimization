package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Raw JSON structures matching the file format

type ProblemJSON struct {
	Widths              []int    `json:"widths"`
	Heights             []int    `json:"heights"`
	Inputs              [][]int  `json:"inputs"`
	Outputs             [][]int  `json:"outputs"`
	BaseCosts           []int64  `json:"base_costs"`
	OpTypes             []string `json:"op_types"`
	FastMemoryCapacity  int64    `json:"fast_memory_capacity"`
	SlowMemoryBandwidth int64    `json:"slow_memory_bandwidth"`
	NativeGranularity   [2]int   `json:"native_granularity"`
}

type SolutionJSON struct {
	Subgraphs         [][]int   `json:"subgraphs"`
	Granularities     [][3]int  `json:"granularities"`
	TensorsToRetain   [][]int   `json:"tensors_to_retain"`
	TraversalOrders   []*[]int  `json:"traversal_orders"` // pointer so we can emit null
	SubgraphLatencies []float64 `json:"subgraph_latencies"`
}

func ReadProblem(filename string) (*Problem, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("reading problem file: %w", err)
	}

	var pj ProblemJSON
	if err := json.Unmarshal(data, &pj); err != nil {
		return nil, fmt.Errorf("parsing problem JSON: %w", err)
	}

	// Build tensors from parallel width/height arrays
	numTensors := len(pj.Widths)
	tensors := make([]Tensor, numTensors)
	for i := 0; i < numTensors; i++ {
		tensors[i] = Tensor{Width: pj.Widths[i], Height: pj.Heights[i]}
	}

	// Build ops from parallel arrays
	numOps := len(pj.Inputs)
	ops := make([]Op, numOps)
	for i := 0; i < numOps; i++ {
		ops[i] = Op{
			OpType:   pj.OpTypes[i],
			Inputs:   pj.Inputs[i],
			Outputs:  pj.Outputs[i],
			BaseCost: pj.BaseCosts[i],
		}
	}

	return &Problem{
		Tensors:             tensors,
		Ops:                 ops,
		FastMemoryCapacity:  pj.FastMemoryCapacity,
		SlowMemoryBandwidth: pj.SlowMemoryBandwidth,
		NativeGranularity:   pj.NativeGranularity,
	}, nil
}

func WriteSolution(filename string, sol *Solution) error {
	sj := SolutionJSON{
		Subgraphs:         make([][]int, len(sol.Subgraphs)),
		Granularities:     make([][3]int, len(sol.Subgraphs)),
		TensorsToRetain:   make([][]int, len(sol.Subgraphs)),
		TraversalOrders:   make([]*[]int, len(sol.Subgraphs)),
		SubgraphLatencies: make([]float64, len(sol.Subgraphs)),
	}

	for i, sg := range sol.Subgraphs {
		sj.Subgraphs[i] = sg.Ops
		sj.Granularities[i] = sg.Granularity
		// Ensure we write empty arrays, not null
		if sg.TensorsToRetain == nil {
			sj.TensorsToRetain[i] = []int{}
		} else {
			sj.TensorsToRetain[i] = sg.TensorsToRetain
		}
		if len(sg.TraversalOrder) > 0 {
			order := make([]int, len(sg.TraversalOrder))
			copy(order, sg.TraversalOrder)
			sj.TraversalOrders[i] = &order
		} else {
			sj.TraversalOrders[i] = nil // emit JSON null
		}
		sj.SubgraphLatencies[i] = sg.SubgraphLatency
	}

	data, err := json.MarshalIndent(sj, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling solution: %w", err)
	}

	return os.WriteFile(filename, data, 0644)
}
