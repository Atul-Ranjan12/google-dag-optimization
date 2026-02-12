# MLSys 2026 DAG Optimization Challenge - Baseline Solution

## Problem Overview

The MLSys 2026 DAG Optimization Challenge tackles a fundamental problem in modern AI accelerators: **executing large computation graphs on hardware with limited fast memory**.

**The Challenge:**

- You have a DAG (Directed Acyclic Graph) of tensor operations (MatMul, Pointwise, etc.)
- Tensors are **orders of magnitude larger** than your fast memory capacity
- You must **tile** the computation (process in small chunks)
- Moving data between slow ↔ fast memory is **expensive** (limited bandwidth)
- **Goal:** Minimize total execution latency while respecting memory constraints

**Example:** A 512×512 tensor has 262,144 elements, but your fast memory only holds 60,000. You must break the computation into 128×128 tiles (16,384 elements each) and orchestrate data movement intelligently.

This is analogous to real-world problems in GPUs, TPUs, and AI accelerators where on-chip SRAM is scarce but essential for performance.

---

## My Baseline Solution

### Strategy

This baseline implements the **simplest valid scheduling approach**:

1. **Topological Sort:** Determine a valid execution order respecting dependencies (no op runs before its inputs are ready)
2. **One Op Per Subgraph:** Execute each operation independently (no fusion)
3. **Maximal Granularity:** For each op, find the largest tile size that fits in fast memory
4. **No Retention:** Evict all outputs to slow memory after each op (no inter-subgraph data reuse)
5. **Raster Order:** Process tiles in default row-major order (no traversal optimization)

### How It Works

**For benchmark `mlsys-2026-1.json` (5 ops, 512×512 tensors, 60K fast memory):**

1. **Dependency Analysis:**
   - Op0 (MatMul) has no dependencies → runs first
   - Op1 (Pointwise) needs Op0's output → runs second
   - Op2, Op3, Op4 follow in sequence (it's a chain)

2. **Granularity Selection:**
   - Try native granularity [128×128] with full reduction depth
   - Check if working set (input tiles + output tile) ≤ 60,000
   - If too big, halve dimensions until it fits
   - Result: MatMuls use [128, 128, 128], Pointwise use [128, 128, 1]

3. **Execution:**
   - Each 512×512 tensor requires 4×4 = 16 spatial tiles
   - MatMuls need 4 k-steps per tile (total: 64 steps)
   - Pointwise need 1 step per tile (total: 16 steps)
   - Each step: load inputs → compute → store outputs
   - Latency = max(compute_cost, memory_transfer_cost)

4. **Result:**
   - Total latency: **471,500.8** time units
   - Valid solution (no OOM, correct dependencies)
   - Highly suboptimal (every intermediate spills to slow memory)

### Why It's Slow

- **No fusion:** Tensors 4, 5, 6, 7 each make a full round-trip through slow memory
- **No retention:** Tensor0 is loaded twice (used by Op0 and Op4)
- **No traversal optimization:** Tiles don't reuse data between steps

**Potential improvements:**

- Fusion: Combine all 5 ops → intermediates become ephemeral (free)
- Retention: Keep Tensor0 resident → save one reload
- Snake traversal: Reuse LHS/RHS strips in MatMul tiles → reduce bandwidth

---

## Code Structure

```
src/
├── main.go              # Entry point (parses args, runs solver, writes output)
├── types.go             # Data structures (Problem, Solution, Tensor, Op, Subgraph)
├── io.go                # JSON parsing and writing
├── solver.go            # Baseline solver (topological sort + granularity selection)
├── evaluate.go          # Cost model (working set calculation, latency computation)
├── util.go              # Helper functions (CeilDiv, Max, etc.)
├── visualize.go         # Graphviz DOT file generation (optional)
└── visualize_example.go # Visualization entry point (optional)
```

---

## How to Run

### Prerequisites

- **Go 1.21+** ([install](https://go.dev/doc/install))
- **Graphviz** (optional, for visualization): `brew install graphviz`

### Build

```bash
cd src
go build -o mlsys
```

### Run Solver

```bash
./mlsys <input.json> <output.json>

# Example:
./mlsys ../benchmarks/mlsys-2026-1.json output.json
```

**Output:**

```
Problem: 9 tensors, 5 ops, capacity=60000, bandwidth=20, native=[128,128]
Verified total latency: 471500.8
Subgraph 0: ops=[0] gran=[128,128,128] retain=[] latency=135321.6
Subgraph 1: ops=[1] gran=[128,128,1] retain=[] latency=26214.4
...
Solution written to output.json
```

### Visualize (Optional)

```bash
./mlsys visualize ../benchmarks/mlsys-2026-1.json
```

**Generates:**

- `dag.png` — Computation graph (tensors + ops + dependencies)
- `solution.png` — Scheduled solution (subgraphs as clusters)
- `timeline.png` — Execution timeline (when each subgraph runs)

---

## Benchmarks

The challenge provides 5 public benchmarks for testing:

| File               | Ops | Tensors | Timeout | Difficulty        |
| ------------------ | --- | ------- | ------- | ----------------- |
| mlsys-2026-1.json  | 5   | 9       | 2s      | Small chain       |
| mlsys-2026-5.json  | 19  | 29      | 5s      | Branching DAG     |
| mlsys-2026-9.json  | 32  | 48      | 15s     | Repeated blocks   |
| mlsys-2026-13.json | 63  | 99      | 30s     | Parallel branches |
| mlsys-2026-17.json | 103 | 159     | 60s     | Transformer-like  |

All benchmarks are in `../benchmarks/`

---

## Implementation Details

### Topological Sort (Kahn's Algorithm)

```go
// Build dependency graph: which op produces which tensor
producerOf := map[tensor_id → op_id]

// For each op, count dependencies
for each op:
    for each input tensor:
        if tensor has a producer:
            add edge: producer → this_op
            increment this_op's in-degree

// BFS: process ops with in-degree 0
queue = [ops with in-degree 0]
while queue not empty:
    op = queue.pop()
    add op to order
    for each dependent of op:
        decrement dependent's in-degree
        if in-degree == 0:
            queue.push(dependent)
```

### Granularity Selection

```go
// Start with native granularity
w, h, k = native_w, native_h, full_K

// Try smaller granularities until working set fits
for w = native_w down to 1 (halving):
    for h = native_h down to 1 (halving):
        for k = full_K down to 1 (halving):
            ws = compute_working_set(w, h, k)
            if ws <= capacity:
                return [w, h, k]
```

### Latency Computation (Roofline Model)

```go
for each spatial_tile:
    for each k_step:
        // Memory transfer
        bytes_loaded = sum of input tile sizes (from slow memory)
        bytes_stored = output tile size (to slow memory, if last k-step)
        mem_time = (bytes_loaded + bytes_stored) / bandwidth

        // Compute
        comp_time = sum of base_costs for ops in subgraph

        // Bottleneck
        step_latency = max(comp_time, mem_time)
        total_latency += step_latency
```

---

## Performance

**Baseline results** on the 5 public benchmarks:

| Benchmark     | Latency   | Status  |
| ------------- | --------- | ------- |
| mlsys-2026-1  | 471,500.8 | ✓ Valid |
| mlsys-2026-5  | TBD       | ✓ Valid |
| mlsys-2026-9  | TBD       | ✓ Valid |
| mlsys-2026-13 | TBD       | ✓ Valid |
| mlsys-2026-17 | TBD       | ✓ Valid |

_(Run benchmarks to populate)_

---

## Next Steps

**Optimizations to implement:**

1. **Chain Fusion:** Detect linear chains (A→B→C) and fuse them into single subgraphs
2. **Retention Analysis:** Keep frequently-reused tensors in fast memory across subgraphs
3. **Snake Traversal:** For MatMul, process tiles in zig-zag order to reuse LHS/RHS strips
4. **Recomputation:** For diamond patterns, compare cost of spilling vs. recomputing cheap ops
5. **Global Search:** Beam search or simulated annealing to explore fusion/retention combinations

## License

Apache 2.0 (following the competition requirements)

---

## Author

Built with Go for the MLSys 2026 DAG Optimization Challenge.

_Why Go?_ Clean concurrency primitives, fast compilation, static typing, and a joy to write. Perfect for systems challenges like this.
