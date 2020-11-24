import numpy as np

# a 2x2 matrix
A = np.array([[0,1],[-2,-3]])

# find eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(A)

# print results
for i, value in enumerate(vals):
    print("Eigenvector:", vecs[:,i], ", Eigenvalue:", value)
    
# Eigenvector: [ 0.70710678 -0.70710678] , Eigenvalue: -1.0
# Eigenvector: [-0.4472136   0.89442719] , Eigenvalue: -2.0