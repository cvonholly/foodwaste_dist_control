import cvxpy as cp
import numpy as np

# Define a cvxpy variable (e.g., a column vector with 3 elements)
y = [cp.Variable((1, 3)) for k in range(3)]
x = y[0]  #cp.Variable((3, 1))
print(x)

# Define a vector (e.g., a column vector with 3 elements)
vector = np.array([[1.0], [2.0], [3.0]])

# Combine using horizontal stacking
matrix = cp.vstack([x, vector.T])

# Print the resulting matrix
print(matrix)
