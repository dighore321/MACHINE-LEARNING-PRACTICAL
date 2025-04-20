import math
import numpy as np
from scipy import linalg

# Math Library
print("Math Library Operations:")
print("Floor of 3.7:", math.floor(3.7))
print("Ceil of 3.2:", math.ceil(3.2))
print("Square root of 16:", math.sqrt(16))
print("Sine of pi/2:", math.sin(math.pi / 2))
print()

# Numpy Array Attributes
array = np.array([[1, 2], [3, 4]])
print("Array:\n", array)
print("Dimensions:", array.ndim)
print("Shape:", array.shape)
print("Size:", array.size)
print("Sum:", np.sum(array))
print("Mean:", np.mean(array))
print("Sorted (flattened):", np.sort(array, axis=None))
print()

# Reshaping arrays
flat_list = [1, 2, 3, 4, 5, 6]
array_1d = np.array(flat_list)
array_2d = array_1d.reshape((2, 3))
array_3d = array_1d.reshape((1, 2, 3))
print("1D Array:", array_1d)
print("2D Array:\n", array_2d)
print("3D Array:\n", array_3d)
print()

# Generating arrays and performing matrix operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[2, 0], [1, 3]])
print("Matrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)
print("Matrix Addition:\n", matrix_a + matrix_b)
print("Matrix Multiplication:\n", np.dot(matrix_a, matrix_b))
print()

# Scipy - Determinant and Eigenvalues
det = linalg.det(matrix_a)
eigenvalues, eigenvectors = linalg.eig(matrix_a)
print("Determinant of Matrix A:", det)
print("Eigenvalues of Matrix A:", eigenvalues)
print("Eigenvectors of Matrix A:\n", eigenvectors)
