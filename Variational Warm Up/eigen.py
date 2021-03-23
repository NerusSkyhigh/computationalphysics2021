# # Eigenvalues and Eigenvectors for Variational Application

import math
import numpy as np
import scipy.linalg as la

#OVERLAP MATRIX BUILDING AND DIAGONALISATION

dim = 4
S = np.empty([dim,dim], dtype = float)

for i in range (0, dim):
    for j in range (0, dim):
        num = 8 * (1+ (-1)**(i + j))
        den = (1 + i + j) * (3 + i + j) * (5 + i + j)
        S[i,j] = num / den

eigvals, eigvecs = la.eig(S)

eigvals = eigvals.real
eigvecs = eigvecs.real

idx = eigvals.argsort()[::1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]

print("This is the overlap matrix\n")
print(S)
print("\nThese are the eigenvalues\n")
print(eigvals)
print("\nThis is the matrix of the eigenvectors\n")
print(eigvecs)


#V MATRIX CONSTRUCTION and TRANSPOSE

V = np.empty([dim,dim], dtype = float)

for i in range (0, dim):
    V[:,i] = (1 / math.sqrt(eigvals[i])) * eigvecs[:,i]

print("This is the V matrix \n")
print(V)

Vtr = V.transpose()

print("\nThis is the V matrix transposed \n")
print(Vtr)

# H MATRIX CONSTRUCTION

H = np.empty([dim,dim], dtype = float)

for i in range (0, dim):
    for j in range (0, dim):
        if i + j > 1:
            num = 4 * (1 + (-1)**(i + j)) * (-1 + i + j + 2 * i * j);
            den = (-1 + i + j) * (1 + i + j) * (3 + i + j);
            H[i,j] = num / den
        elif i + j <= 1 :
            H[i,j] = 0

print("This is the H matrix \n")
print(H)

# MATRIX MULTIPLICATIONS

VtH = np.matmul(Vtr, H)
Hp = np.matmul(VtH, V)

print("This is the H' matrix \n")
print(Hp)


# EIGENVALUES AND EIGENVECTORS

eigenvals, eigenvecs = la.eig(Hp)

eigenvals = eigenvals.real
eigenvecs = eigenvecs.real

idx1 = eigenvals.argsort()[::1]
eigenvals = eigenvals[idx1]
eigenvecs = eigenvecs[:,idx1]

print("These are the eigenvalues\n")
print(eigenvals)
print("\nThis is the matrix of the eigenvectors\n")
print(eigenvecs)
