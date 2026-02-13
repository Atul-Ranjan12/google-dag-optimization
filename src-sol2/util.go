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

func containsInt(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

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

// divisorsOf returns all divisors of n that are >= minVal, sorted ascending
func divisorsOf(n, minVal int) []int {
	var divs []int
	for i := 1; i*i <= n; i++ {
		if n%i == 0 {
			if i >= minVal {
				divs = append(divs, i)
			}
			j := n / i
			if j != i && j >= minVal {
				divs = append(divs, j)
			}
		}
	}
	// sort
	for i := 0; i < len(divs); i++ {
		for j := i + 1; j < len(divs); j++ {
			if divs[j] < divs[i] {
				divs[i], divs[j] = divs[j], divs[i]
			}
		}
	}
	return divs
}

// powersOf2UpTo returns powers of 2 from minVal up to maxVal
func powersOf2UpTo(maxVal, minVal int) []int {
	var result []int
	for v := minVal; v <= maxVal; v *= 2 {
		if v > 0 {
			result = append(result, v)
		}
	}
	return result
}
