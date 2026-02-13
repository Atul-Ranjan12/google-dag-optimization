package main

func CeilDiv(a, b int) int {
	return (a + b - 1) / b
}

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func MaxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func MaxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func MinFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func AbsInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// containsInt checks if a slice contains a value
func containsInt(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// uniqueInts removes duplicates preserving order
func uniqueInts(s []int) []int {
	seen := make(map[int]bool)
	result := make([]int, 0, len(s))
	for _, v := range s {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	return result
}
