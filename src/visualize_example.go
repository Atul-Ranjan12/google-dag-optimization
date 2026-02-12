package main

import (
	"fmt"
	"os"
)

// This is a standalone program to visualize a benchmark.
// Build: go build -o visualize_example visualize_example.go visualize.go types.go io.go util.go evaluate.go solver.go
// Run: ./visualize_example ../benchmarks/mlsys-2026-1.json

func mainVisualize() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input.json>\n", os.Args[0])
		os.Exit(1)
	}

	inputFile := os.Args[1]

	// Read problem
	problem, err := ReadProblem(inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading problem: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("=== Problem Visualization ===\n")
	fmt.Printf("Tensors: %d, Ops: %d\n", len(problem.Tensors), len(problem.Ops))

	// 1. Visualize the original DAG
	fmt.Println("\n1. Generating DAG visualization...")
	if err := VisualizeProblem(problem, "dag.dot", "dag.png"); err != nil {
		fmt.Fprintf(os.Stderr, "Error visualizing problem: %v\n", err)
	} else {
		fmt.Println("   ✓ Saved: dag.png")
	}

	// 2. Solve with baseline
	fmt.Println("\n2. Running baseline solver...")
	solution := SolveBaseline(problem)

	totalLat, err := EvaluateSolution(problem, solution)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Solution validation error: %v\n", err)
	} else {
		fmt.Printf("   ✓ Total latency: %.1f\n", totalLat)
	}

	// 3. Visualize the solution (with subgraph clusters)
	fmt.Println("\n3. Generating solution visualization...")
	if err := VisualizeSolution(problem, solution, "solution.dot", "solution.png"); err != nil {
		fmt.Fprintf(os.Stderr, "Error visualizing solution: %v\n", err)
	} else {
		fmt.Println("   ✓ Saved: solution.png")
	}

	// 4. Visualize execution timeline
	fmt.Println("\n4. Generating execution timeline...")
	if err := VisualizeExecutionTimeline(solution, "timeline.dot", "timeline.png"); err != nil {
		fmt.Fprintf(os.Stderr, "Error visualizing timeline: %v\n", err)
	} else {
		fmt.Println("   ✓ Saved: timeline.png")
	}

	fmt.Println("\n=== Summary ===")
	fmt.Printf("Total subgraphs: %d\n", len(solution.Subgraphs))
	fmt.Printf("Total latency: %.1f\n", totalLat)
	fmt.Println("\nGenerated files:")
	fmt.Println("  - dag.png       (original computation graph)")
	fmt.Println("  - solution.png  (solution with subgraph clusters)")
	fmt.Println("  - timeline.png  (execution timeline)")
}
