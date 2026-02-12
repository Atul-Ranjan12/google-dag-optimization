package main

import (
	"fmt"
	"os"
)

func main() {
	// Check if running in visualization mode
	if len(os.Args) >= 2 && os.Args[1] == "visualize" {
		// Remove "visualize" from args so visualize_example can parse normally
		os.Args = append(os.Args[:1], os.Args[2:]...)
		mainVisualize()
		return
	}

	// Normal solver mode
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  Solver mode:  %s <input.json> <output.json>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Visualize:    %s visualize <input.json>\n", os.Args[0])
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]

	// Read the problem
	problem, err := ReadProblem(inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading problem: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Problem: %d tensors, %d ops, capacity=%d, bandwidth=%d, native=[%d,%d]\n",
		len(problem.Tensors), len(problem.Ops),
		problem.FastMemoryCapacity, problem.SlowMemoryBandwidth,
		problem.NativeGranularity[0], problem.NativeGranularity[1])

	// Solve
	solution := SolveBaseline(problem)

	// Verify
	totalLat, err := EvaluateSolution(problem, solution)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Solution validation error: %v\n", err)
	} else {
		fmt.Printf("Verified total latency: %.1f\n", totalLat)
	}

	// Print summary
	PrintSolutionSummary(problem, solution)

	// Write output
	if err := WriteSolution(outputFile, solution); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing solution: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Solution written to %s\n", outputFile)
}
