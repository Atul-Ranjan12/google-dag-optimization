package main

// Tensor represents a 2D matrix in the computation graph.
type Tensor struct {
	Width  int
	Height int
}

// Op represents one operation in the DAG.
type Op struct {
	OpType   string
	Inputs   []int
	Outputs  []int
	BaseCost int64
}

// Problem is the full input specification.
type Problem struct {
	Tensors             []Tensor
	Ops                 []Op
	FastMemoryCapacity  int64
	SlowMemoryBandwidth int64
	NativeGranularity   [2]int
}

// Subgraph is one step in our execution schedule.
type Subgraph struct {
	Ops             []int
	Granularity     [3]int
	TensorsToRetain []int
	TraversalOrder  []int
	SubgraphLatency float64
}

// Solution is the full output.
type Solution struct {
	Subgraphs []Subgraph
}
