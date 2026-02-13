package main

import (
	"fmt"
	"math"
	"time"
)

// SubgraphCandidate represents a group of ops we might fuse
type SubgraphCandidate struct {
	Ops         []int
	Granularity [3]int
	Latency     float64
	Feasible    bool
}

// FusionContext tracks fusion state to prevent infinite loops
type FusionContext struct {
	StartTime    time.Time
	TimeoutSec   int
	MaxDepth     int
	CurrentDepth int
}

func NewFusionContext() *FusionContext {
	return &FusionContext{
		StartTime:  time.Now(),
		TimeoutSec: 30, // 30 second timeout per chain
		MaxDepth:   5,  // Max recursion depth
	}
}

func (fc *FusionContext) ShouldStop() bool {
	return time.Since(fc.StartTime).Seconds() > float64(fc.TimeoutSec) || fc.CurrentDepth > fc.MaxDepth
}

// TryFuseChainSmart attempts to fuse a chain with cost-based decisions
func TryFuseChainSmart(p *Problem, chain []int, residentTensors map[int]bool) []SubgraphCandidate {
	ctx := NewFusionContext()
	return tryFuseChainWithContext(p, chain, residentTensors, ctx)
}

func tryFuseChainWithContext(p *Problem, chain []int, residentTensors map[int]bool, ctx *FusionContext) []SubgraphCandidate {
	if ctx.ShouldStop() {
		// Timeout - fall back to no fusion
		fmt.Printf("    [fusion timeout, splitting chain of %d ops]\n", len(chain))
		return noFusionFallback(p, chain, residentTensors)
	}

	ctx.CurrentDepth++
	defer func() { ctx.CurrentDepth-- }()

	if len(chain) == 1 {
		// Single op - just find best granularity
		gran := FindBestGranularity(p, chain, residentTensors)
		lat := EvaluateSubgraphSimple(p, chain, gran, nil, residentTensors)
		return []SubgraphCandidate{
			{Ops: chain, Granularity: gran, Latency: lat, Feasible: true},
		}
	}

	// For long chains, use a simpler heuristic
	if len(chain) > 8 {
		return splitLongChain(p, chain, residentTensors)
	}

	// Try full chain fusion first
	fullGran := FindBestGranularity(p, chain, residentTensors)
	fullLat := EvaluateSubgraphSimple(p, chain, fullGran, nil, residentTensors)
	fullWS := ComputeWorkingSet(p, chain, fullGran, residentTensors)

	// Check if full fusion is reasonable
	fusionReasonable := fullWS <= p.FastMemoryCapacity && IsGranularityReasonable(p, fullGran)

	if !fusionReasonable {
		// Full fusion forces unreasonable granularity - split it
		return splitChainBinary(p, chain, residentTensors, ctx)
	}

	// Compare fused cost vs baseline (no fusion)
	baselineLat := estimateBaselineLatency(p, chain, residentTensors)

	// If fusion is significantly better (>10% improvement), use it
	if fullLat < baselineLat*0.90 {
		return []SubgraphCandidate{
			{Ops: chain, Granularity: fullGran, Latency: fullLat, Feasible: true},
		}
	}

	// Fusion not beneficial enough - try binary split
	splitCandidates := splitChainBinary(p, chain, residentTensors, ctx)

	// Compare split vs no fusion
	splitLat := 0.0
	for _, cand := range splitCandidates {
		splitLat += cand.Latency
	}

	if splitLat < baselineLat*0.95 {
		return splitCandidates
	}

	// No fusion is best
	return noFusionFallback(p, chain, residentTensors)
}

// splitLongChain uses a simple heuristic for very long chains
func splitLongChain(p *Problem, chain []int, residentTensors map[int]bool) []SubgraphCandidate {
	// Split into chunks of 4 ops max
	chunkSize := 4
	var candidates []SubgraphCandidate

	for i := 0; i < len(chain); i += chunkSize {
		end := MinInt(i+chunkSize, len(chain))
		chunk := chain[i:end]

		gran := FindBestGranularity(p, chunk, residentTensors)
		lat := EvaluateSubgraphSimple(p, chunk, gran, nil, residentTensors)

		candidates = append(candidates, SubgraphCandidate{
			Ops:         chunk,
			Granularity: gran,
			Latency:     lat,
			Feasible:    true,
		})
	}

	return candidates
}

// splitChainBinary tries binary split only
func splitChainBinary(p *Problem, chain []int, residentTensors map[int]bool, ctx *FusionContext) []SubgraphCandidate {
	if len(chain) <= 1 {
		gran := FindBestGranularity(p, chain, residentTensors)
		lat := EvaluateSubgraphSimple(p, chain, gran, nil, residentTensors)
		return []SubgraphCandidate{
			{Ops: chain, Granularity: gran, Latency: lat, Feasible: true},
		}
	}

	// Try only the midpoint split
	mid := len(chain) / 2
	left := chain[:mid]
	right := chain[mid:]

	leftSub := tryFuseChainWithContext(p, left, residentTensors, ctx)
	rightSub := tryFuseChainWithContext(p, right, residentTensors, ctx)

	return append(leftSub, rightSub...)
}

// noFusionFallback creates individual subgraphs for each op
func noFusionFallback(p *Problem, chain []int, residentTensors map[int]bool) []SubgraphCandidate {
	var candidates []SubgraphCandidate
	for _, opIdx := range chain {
		gran := FindBestGranularity(p, []int{opIdx}, residentTensors)
		lat := EvaluateSubgraphSimple(p, []int{opIdx}, gran, nil, residentTensors)
		candidates = append(candidates, SubgraphCandidate{
			Ops:         []int{opIdx},
			Granularity: gran,
			Latency:     lat,
			Feasible:    true,
		})
	}
	return candidates
}

// estimateBaselineLatency estimates cost of no fusion
func estimateBaselineLatency(p *Problem, chain []int, residentTensors map[int]bool) float64 {
	var baselineLat float64
	for _, opIdx := range chain {
		singleGran := FindBestGranularity(p, []int{opIdx}, residentTensors)
		singleLat := EvaluateSubgraphSimple(p, []int{opIdx}, singleGran, nil, residentTensors)
		baselineLat += singleLat
	}

	// Add penalty for memory transfers between subgraphs
	transferPenalty := estimateTransferPenalty(p, chain)
	baselineLat += transferPenalty

	return baselineLat
}

// estimateTransferPenalty estimates the cost of memory transfers in non-fused execution
func estimateTransferPenalty(p *Problem, chain []int) float64 {
	penalty := 0.0
	bw := float64(p.SlowMemoryBandwidth)

	for i := 0; i < len(chain)-1; i++ {
		op := p.Ops[chain[i]]
		for _, outTensor := range op.Outputs {
			t := p.Tensors[outTensor]
			transferSize := int64(t.Width) * int64(t.Height)
			// Transfer cost: evict + reload = 2x transfer
			penalty += 2.0 * float64(transferSize) / bw
		}
	}

	return penalty
}

// FormSubgraphs creates the initial subgraph formation via intelligent fusion
func FormSubgraphs(p *Problem, gi *GraphInfo) []SubgraphCandidate {
	chains := FindLinearChains(p, gi)

	fmt.Printf("  Found %d chains to process\n", len(chains))

	var candidates []SubgraphCandidate
	residentTensors := make(map[int]bool)

	for i, chain := range chains {
		fmt.Printf("  Processing chain %d/%d (length=%d)...\n", i+1, len(chains), len(chain))
		subs := TryFuseChainSmart(p, chain, residentTensors)
		fmt.Printf("    -> Created %d subgraphs\n", len(subs))
		candidates = append(candidates, subs...)
	}

	return candidates
}

// OptimizeSubgraphGranularity refines the granularity with traversal consideration
func OptimizeSubgraphGranularity(p *Problem, ops []int, residentTensors map[int]bool) (gran [3]int, traversal []int, lat float64) {
	// Find best granularity
	gran = FindBestGranularity(p, ops, residentTensors)

	// Determine if we should use snake traversal
	hasMatMul := false
	for _, opIdx := range ops {
		if p.Ops[opIdx].OpType == "MatMul" {
			hasMatMul = true
			break
		}
	}

	if hasMatMul {
		// Compute traversal
		primaryOutput := GetOutputTensor(p, ops)
		outT := p.Tensors[primaryOutput]
		nCols := CeilDiv(outT.Width, gran[0])
		nRows := CeilDiv(outT.Height, gran[1])

		if nCols*nRows > 1 {
			// Use snake
			traversal = SnakeTraversal(nCols, nRows)
		}
	}

	// Evaluate with detailed model
	lat, err := EvaluateSubgraphDetailed(p, ops, gran, nil, traversal, residentTensors)
	if err != nil {
		lat = math.Inf(1)
	}

	return gran, traversal, lat
}
