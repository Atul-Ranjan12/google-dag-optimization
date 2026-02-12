package main

// CeilDiv returns ceil(a / b) for positive integers.
// Example: CeilDiv(512, 128) = 4, CeilDiv(513, 128) = 5
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
