# Practical 1: Demonstrating usage of NumPy, SciPy, Math, and Statistics

import numpy as np
import scipy.special as sp
from math import sqrt, sin, ceil, floor, exp, pi
import statistics

# Sample numerical data
data = [10, 20, 30, 40, 50]

print("Original Data:", data)

# NumPy Operations
np_array = np.array(data)
print("\n--- NumPy ---")
print("Array:", np_array)
print("Mean (NumPy):", np.mean(np_array))
print("Standard Deviation (NumPy):", np.std(np_array))
print("Sum (NumPy):", np.sum(np_array))
print("Square Root (element-wise):", np.sqrt(np_array))

# Math Library Operations
print("\n--- Math ---")
print("Square root of 16:", sqrt(16))
print("Exponential of 2:", exp(2))
print("Sine of π/2:", sin(pi/2))
print("Floor of 4.7:", floor(4.7))
print("Ceil of 4.3:", ceil(4.3))

# Statistics Module
print("\n--- Statistics ---")
print("Mean (Statistics):", statistics.mean(data))
print("Median:", statistics.median(data))
print("Variance:", statistics.variance(data))
print("Standard Deviation:", statistics.stdev(data))

# SciPy Special Functions
print("\n--- SciPy ---")
print("Gamma(5):", sp.gamma(5))
print("Factorial(5):", sp.factorial(5))
print("Log Gamma(5):", sp.gammaln(5))
