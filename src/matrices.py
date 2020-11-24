import numpy as np

# Adjacency matrix
A = np.array([
  [0, 1, 1, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [1, 1, 0, 1, 0, 0],
  [0, 0, 1, 0, 1, 1],
  [0, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 1, 0]])

# Degree matrix
D = np.diag(A.sum(axis=1))

# Graph Laplacian
L = D - A

# Eigenvalues, Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L)

# Sort eigenvalues/vectors by ascending order of eigenvalue
eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
eigenvalues = eigenvalues[np.argsort(eigenvalues)]

print(eigenvalues)