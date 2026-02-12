package main

// Tensor represents a 2D matrix in the computation graph.
// Width = number of columns, Height = number of rows.
// Size in "memory units" = Width * Height.
type Tensor struct {
	Width  int
	Height int
}

// Op represents one operation in the DAG.
// It consumes input tensors and produces output tensors.
// For MatMul, inputs[0] is LHS, inputs[1] is RHS.
// Output = LHS(Height x K) @ RHS(K x Width) => Output(Height x Width)
// For Pointwise, all inputs and outputs share the same shape.
type Op struct {
	OpType   string // "MatMul" or "Pointwise"
	Inputs   []int  // indices into the tensor list
	Outputs  []int  // indices into the tensor list
	BaseCost int64  // compute cost at native granularity
}

// Problem is the full input specification.
type Problem struct {
	Tensors             []Tensor
	Ops                 []Op
	FastMemoryCapacity  int64
	SlowMemoryBandwidth int64
	NativeGranularity   [2]int // [width, height]
}

// Subgraph is one step in our execution schedule.
type Subgraph struct {
	Ops             []int   // which op indices to execute
	Granularity     [3]int  // [w, h, k] tile size
	TensorsToRetain []int   // output tensor indices to keep in fast memory
	TraversalOrder  []int   // nil/empty means default raster order
	SubgraphLatency float64 // computed total latency for this subgraph
}

// Solution is the full output — an ordered list of subgraphs.
type Solution struct {
	Subgraphs []Subgraph
}
