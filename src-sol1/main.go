package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	benchmarkDir := "../benchmarks"
	outputDir := "./solutions"

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Find all benchmark files
	files, err := filepath.Glob(filepath.Join(benchmarkDir, "mlsys-2026-*.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error finding benchmark files: %v\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "No benchmark files found in %s\n", benchmarkDir)
		os.Exit(1)
	}

	fmt.Println("=" + strings.Repeat("=", 78))
	fmt.Println("  MLSys 2026 DAG Optimization - Optimized Solver")
	fmt.Println("=" + strings.Repeat("=", 78))
	fmt.Printf("Found %d benchmark files\n\n", len(files))

	results := make([]BenchmarkResult, 0, len(files))

	for i, inputFile := range files {
		baseName := filepath.Base(inputFile)
		benchmarkName := strings.TrimSuffix(baseName, ".json")
		outputFile := filepath.Join(outputDir, benchmarkName+"-solution.json")

		fmt.Printf("[%d/%d] Processing: %s\n", i+1, len(files), baseName)
		fmt.Println(strings.Repeat("-", 80))

		startTime := time.Now()

		// Read problem
		problem, err := ReadProblem(inputFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  ✗ Error reading problem: %v\n\n", err)
			continue
		}

		fmt.Printf("  Problem: %d tensors, %d ops, capacity=%d, bandwidth=%d\n",
			len(problem.Tensors), len(problem.Ops),
			problem.FastMemoryCapacity, problem.SlowMemoryBandwidth)

		// Solve
		solution := SolveOptimized(problem)

		// Verify
		totalLat, err := EvaluateSolution(problem, solution)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  ✗ Solution validation error: %v\n", err)
			// Try to recover
			for i := range solution.Subgraphs {
				sg := &solution.Subgraphs[i]
				resident := make(map[int]bool)
				if i > 0 {
					for _, t := range solution.Subgraphs[i-1].TensorsToRetain {
						resident[t] = true
					}
				}
				lat, _ := EvaluateSubgraphDetailed(
					problem, sg.Ops, sg.Granularity, sg.TensorsToRetain,
					sg.TraversalOrder, resident,
				)
				sg.SubgraphLatency = lat
			}
			totalLat, _ = EvaluateSolution(problem, solution)
		}

		elapsed := time.Since(startTime)

		// Write output
		if err := WriteSolution(outputFile, solution); err != nil {
			fmt.Fprintf(os.Stderr, "  ✗ Error writing solution: %v\n\n", err)
			continue
		}

		fmt.Printf("  ✓ Total Latency: %.1f\n", totalLat)
		fmt.Printf("  ✓ Subgraphs: %d\n", len(solution.Subgraphs))
		fmt.Printf("  ✓ Time: %v\n", elapsed)
		fmt.Printf("  ✓ Output: %s\n", outputFile)

		results = append(results, BenchmarkResult{
			Name:      benchmarkName,
			Latency:   totalLat,
			Subgraphs: len(solution.Subgraphs),
			Time:      elapsed,
		})

		fmt.Println()
	}

	// Print summary
	fmt.Println("=" + strings.Repeat("=", 78))
	fmt.Println("  SUMMARY")
	fmt.Println("=" + strings.Repeat("=", 78))
	fmt.Printf("%-30s %15s %12s %12s\n", "Benchmark", "Latency", "Subgraphs", "Time")
	fmt.Println(strings.Repeat("-", 80))

	for _, result := range results {
		fmt.Printf("%-30s %15.1f %12d %12v\n",
			result.Name, result.Latency, result.Subgraphs, result.Time)
	}

	fmt.Println("=" + strings.Repeat("=", 78))
	fmt.Printf("Total benchmarks completed: %d/%d\n", len(results), len(files))
	fmt.Println("=" + strings.Repeat("=", 78))
}

type BenchmarkResult struct {
	Name      string
	Latency   float64
	Subgraphs int
	Time      time.Duration
}
