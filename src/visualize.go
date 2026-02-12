package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// checkGraphviz verifies that the 'dot' command is available
func checkGraphviz() error {
	cmd := exec.Command("which", "dot")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("graphviz 'dot' command not found. Please install: brew install graphviz")
	}
	return nil
}

// renderDotToPNG converts a .dot file to .png using Graphviz
func renderDotToPNG(dotFile, pngFile string) error {
	if err := checkGraphviz(); err != nil {
		return err
	}

	cmd := exec.Command("dot", "-Tpng", dotFile, "-o", pngFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("graphviz error: %w\nOutput: %s", err, string(output))
	}

	// Verify the PNG was created
	if _, err := os.Stat(pngFile); os.IsNotExist(err) {
		return fmt.Errorf("PNG file was not created: %s", pngFile)
	}

	return nil
}

// VisualizeProblem generates a Graphviz DOT file and renders it as PNG.
func VisualizeProblem(p *Problem, dotFile, pngFile string) error {
	// Identify graph inputs and outputs
	producedBy := make(map[int]int)
	consumedBy := make(map[int][]int)

	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			producedBy[t] = i
		}
		for _, t := range op.Inputs {
			consumedBy[t] = append(consumedBy[t], i)
		}
	}

	graphInputs := make(map[int]bool)
	graphOutputs := make(map[int]bool)

	for i := range p.Tensors {
		if _, produced := producedBy[i]; !produced {
			graphInputs[i] = true
		}
		if len(consumedBy[i]) == 0 {
			graphOutputs[i] = true
		}
	}

	// Build DOT content
	var sb strings.Builder
	sb.WriteString("digraph DAG {\n")
	sb.WriteString("  rankdir=TB;\n")
	sb.WriteString("  node [shape=box, style=rounded, fontname=\"Arial\"];\n")
	sb.WriteString("  edge [fontname=\"Arial\", fontsize=10];\n\n")

	// Draw tensors
	for i, t := range p.Tensors {
		color := "white"
		label := fmt.Sprintf("Tensor[%d]\\n%dx%d", i, t.Width, t.Height)

		if graphInputs[i] {
			color = "lightgreen"
			label += "\\n(input)"
		} else if graphOutputs[i] {
			color = "lightblue"
			label += "\\n(output)"
		}

		sb.WriteString(fmt.Sprintf("  T%d [label=\"%s\", fillcolor=\"%s\", style=\"rounded,filled\"];\n",
			i, label, color))
	}

	sb.WriteString("\n")

	// Draw ops
	for i, op := range p.Ops {
		label := fmt.Sprintf("Op[%d]\\n%s\\ncost=%d", i, op.OpType, op.BaseCost)
		sb.WriteString(fmt.Sprintf("  Op%d [label=\"%s\", shape=box, fillcolor=\"lightyellow\", style=\"filled\"];\n",
			i, label))
	}

	sb.WriteString("\n")

	// Draw edges: tensor -> op (inputs)
	for i, op := range p.Ops {
		for pos, t := range op.Inputs {
			edgeLabel := ""
			if op.OpType == "MatMul" {
				if pos == 0 {
					edgeLabel = "[label=\"LHS\"]"
				} else {
					edgeLabel = "[label=\"RHS\"]"
				}
			}
			sb.WriteString(fmt.Sprintf("  T%d -> Op%d %s;\n", t, i, edgeLabel))
		}
	}

	sb.WriteString("\n")

	// Draw edges: op -> tensor (outputs)
	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			sb.WriteString(fmt.Sprintf("  Op%d -> T%d;\n", i, t))
		}
	}

	sb.WriteString("}\n")

	// Write DOT file
	if err := os.WriteFile(dotFile, []byte(sb.String()), 0644); err != nil {
		return fmt.Errorf("writing DOT file: %w", err)
	}

	fmt.Printf("   ✓ Created DOT file: %s\n", dotFile)

	// Render to PNG
	if err := renderDotToPNG(dotFile, pngFile); err != nil {
		fmt.Printf("   ⚠ Could not render PNG: %v\n", err)
		fmt.Printf("   → You can view the DOT file at: https://dreampuf.github.io/GraphvizOnline/\n")
		fmt.Printf("   → Or manually convert: dot -Tpng %s -o %s\n", dotFile, pngFile)
		return err
	}

	return nil
}

// VisualizeSolution shows the execution schedule with subgraph boundaries.
func VisualizeSolution(p *Problem, sol *Solution, dotFile, pngFile string) error {
	// Identify graph inputs and outputs
	producedBy := make(map[int]int)
	consumedBy := make(map[int][]int)

	for i, op := range p.Ops {
		for _, t := range op.Outputs {
			producedBy[t] = i
		}
		for _, t := range op.Inputs {
			consumedBy[t] = append(consumedBy[t], i)
		}
	}

	graphInputs := make(map[int]bool)
	graphOutputs := make(map[int]bool)

	for i := range p.Tensors {
		if _, produced := producedBy[i]; !produced {
			graphInputs[i] = true
		}
		if len(consumedBy[i]) == 0 {
			graphOutputs[i] = true
		}
	}

	// Map ops to subgraphs
	opToSubgraph := make(map[int]int)
	for sgIdx, sg := range sol.Subgraphs {
		for _, opIdx := range sg.Ops {
			opToSubgraph[opIdx] = sgIdx
		}
	}

	var sb strings.Builder
	sb.WriteString("digraph Solution {\n")
	sb.WriteString("  rankdir=TB;\n")
	sb.WriteString("  node [fontname=\"Arial\"];\n")
	sb.WriteString("  edge [fontname=\"Arial\", fontsize=10];\n\n")

	// Draw tensors (not in clusters)
	for i, t := range p.Tensors {
		color := "white"
		label := fmt.Sprintf("T[%d]\\n%dx%d", i, t.Width, t.Height)

		if graphInputs[i] {
			color = "lightgreen"
		} else if graphOutputs[i] {
			color = "lightblue"
		}

		sb.WriteString(fmt.Sprintf("  T%d [label=\"%s\", shape=ellipse, fillcolor=\"%s\", style=\"filled\"];\n",
			i, label, color))
	}

	sb.WriteString("\n")

	// Draw subgraphs as clusters
	for sgIdx, sg := range sol.Subgraphs {
		sb.WriteString(fmt.Sprintf("  subgraph cluster_%d {\n", sgIdx))
		sb.WriteString(fmt.Sprintf("    label=\"Subgraph %d\\nGran=[%d,%d,%d]\\nLatency=%.1f\";\n",
			sgIdx, sg.Granularity[0], sg.Granularity[1], sg.Granularity[2], sg.SubgraphLatency))
		sb.WriteString("    style=filled;\n")
		sb.WriteString("    color=lightgrey;\n")
		sb.WriteString("    node [style=filled, fillcolor=lightyellow];\n\n")

		for _, opIdx := range sg.Ops {
			op := p.Ops[opIdx]
			label := fmt.Sprintf("Op[%d]\\n%s\\ncost=%d", opIdx, op.OpType, op.BaseCost)
			sb.WriteString(fmt.Sprintf("    Op%d [label=\"%s\"];\n", opIdx, label))
		}

		sb.WriteString("  }\n\n")
	}

	// Draw edges
	for i, op := range p.Ops {
		for pos, t := range op.Inputs {
			edgeLabel := ""
			if op.OpType == "MatMul" {
				if pos == 0 {
					edgeLabel = "[label=\"LHS\"]"
				} else {
					edgeLabel = "[label=\"RHS\"]"
				}
			}
			sb.WriteString(fmt.Sprintf("  T%d -> Op%d %s;\n", t, i, edgeLabel))
		}

		for _, t := range op.Outputs {
			sb.WriteString(fmt.Sprintf("  Op%d -> T%d;\n", i, t))
		}
	}

	// Highlight retained tensors
	retainedInAnySubgraph := make(map[int]bool)
	for _, sg := range sol.Subgraphs {
		for _, t := range sg.TensorsToRetain {
			retainedInAnySubgraph[t] = true
		}
	}

	if len(retainedInAnySubgraph) > 0 {
		sb.WriteString("\n  // Retained tensors\n")
		for t := range retainedInAnySubgraph {
			sb.WriteString(fmt.Sprintf("  T%d [penwidth=3, color=red];\n", t))
		}
	}

	sb.WriteString("}\n")

	// Write DOT file
	if err := os.WriteFile(dotFile, []byte(sb.String()), 0644); err != nil {
		return fmt.Errorf("writing DOT file: %w", err)
	}

	fmt.Printf("   ✓ Created DOT file: %s\n", dotFile)

	// Render to PNG
	if err := renderDotToPNG(dotFile, pngFile); err != nil {
		fmt.Printf("   ⚠ Could not render PNG: %v\n", err)
		fmt.Printf("   → You can view the DOT file at: https://dreampuf.github.io/GraphvizOnline/\n")
		return err
	}

	return nil
}

// VisualizeExecutionTimeline creates a horizontal timeline showing when each
// subgraph executes and how long it takes.
func VisualizeExecutionTimeline(sol *Solution, dotFile, pngFile string) error {
	var sb strings.Builder
	sb.WriteString("digraph Timeline {\n")
	sb.WriteString("  rankdir=LR;\n")
	sb.WriteString("  node [shape=box, fontname=\"Arial\"];\n\n")

	currentTime := 0.0

	for i, sg := range sol.Subgraphs {
		endTime := currentTime + sg.SubgraphLatency
		label := fmt.Sprintf("SG%d\\nOps=%v\\nTime: %.1f - %.1f\\n(%.1f units)",
			i, sg.Ops, currentTime, endTime, sg.SubgraphLatency)

		sb.WriteString(fmt.Sprintf("  SG%d [label=\"%s\", fillcolor=\"lightblue\", style=\"filled\"];\n",
			i, label))

		if i > 0 {
			sb.WriteString(fmt.Sprintf("  SG%d -> SG%d [label=\"\"];\n", i-1, i))
		}

		currentTime = endTime
	}

	sb.WriteString(fmt.Sprintf("\n  Total [label=\"Total Latency:\\n%.1f\", shape=ellipse, fillcolor=\"lightgreen\", style=\"filled\"];\n",
		currentTime))
	sb.WriteString(fmt.Sprintf("  SG%d -> Total;\n", len(sol.Subgraphs)-1))

	sb.WriteString("}\n")

	if err := os.WriteFile(dotFile, []byte(sb.String()), 0644); err != nil {
		return fmt.Errorf("writing DOT file: %w", err)
	}

	fmt.Printf("   ✓ Created DOT file: %s\n", dotFile)

	if err := renderDotToPNG(dotFile, pngFile); err != nil {
		fmt.Printf("   ⚠ Could not render PNG: %v\n", err)
		fmt.Printf("   → You can view the DOT file at: https://dreampuf.github.io/GraphvizOnline/\n")
		return err
	}

	return nil
}
