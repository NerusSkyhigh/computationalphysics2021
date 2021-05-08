import math as m
import numpy as np
#import matplotlib.pyplot as plt
from numba import njit
import scipy.linalg as la

Z = 2 # For Helium
alphas = np.array([14.899983, 2.726485, 0.757447, 0.251390])

a = 0.001
delta = 0.001


@njit
def OneBodyH(alphas):
    # Compute the one body terms of the Hamiltonian
    H_mat = np.zeros([4,4])

    for i in range(0, alphas.shape[0]):
        for j in range(0, i+1):

            Kinetic = 3 * (alphas[i] * alphas[j] * m.pi**(3.0/2.0) ) / \
                         ( (alphas[i]+alphas[j])**(5.0/2.0) )

            Vext = - Z * 2 * m.pi / (alphas[i] + alphas[j])

            H_mat[i,j] = H_mat[j,i] = 2*Kinetic + 2*Vext

    return H_mat



@njit
def OverlapMatrix(alphas):
    # Compute the overlap Matrix
    S = np.zeros([4,4])

    for i in range(0, alphas.shape[0]):
        for j in range(0, i+1):
            S[i,j] = S[j,i] = (m.pi / (alphas[i] + alphas[j] ) )**(3.0/2.0)

    return S


# Computes the direct and exchange integrals with the given orbitals
@njit
def integrals(alphas):
    # 2-BODY POTENTIAL
    Pot_mat = np.zeros((16,16))

    for p in range(0, 4):
        for q in range(0, p+1):
            for r in range(0, 4):
                for s in range(0, r+1):

                    if (4*s+r <= 4*q+p):
                        Pot_mat[4*q+p, 4*s+r] = m.pi**(5.0/2.0) / (4 * alphas[p]
                        * alphas[q] * alphas[r] * alphas[s]) * (m.sqrt(alphas[p] + alphas[q])
                        * m.sqrt(alphas[r] + alphas[s]) / m.sqrt(alphas[p] + alphas[q] + alphas[r]
                        + alphas[s])) * S[p,q] * S[r,s]


    # Now we must assign a value to the missing elements in the "upper part"
    for i in range(0,16):
        j = 0
        while j < i:
            Pot_mat[j,i] = Pot_mat[i,j]
            j += 1


    # Now fill the missing rows and columns
    for q in range (0,4,1):
        for p in range (q,4,1):
            Pot_mat[(4*p +q), :] = Pot_mat[(4*q + p), :]


    for q in range (0,4,1):
        for p in range (q,4,1):
            Pot_mat[:, (4*p +q)] = Pot_mat[:, (4*q + p)]


    return Pot_mat


# Building the Fock Matrix, representing the Fock operator PROBLEMS

#@njit
def fock(H, P, C, C_old):

    F = np.zeros((4,4))

    for p in range(0,4,1):
        for q in range(0,4,1):

            twob = 0.0

            for r in range(0,4,1):
                for s in range(0,4,1):

                    twob += a * C[r] * C[s] * ( P[4*q + p, 4*s + r] - P[4*s +p, 4*q +r] )
                    + (1 - a) * C_old[r] * C_old[s] * ( P[4*q + p, 4*s + r] - P[4*s +p, 4*q +r] )

            F[p,q] = H[p,q] + twob


    return F


# Now thefunction to diagonalise the generalised eigenvalue problem
def diagonalisation(F, S):

    # Diagonalise S matrix

    eigvalS, eigvecS = la.eig(S)

    eigvalS = eigvalS.real
    eigvecS = eigvecS.real

    # Sort them

    idx = eigvalS.argsort()[::1]
    eigvalS = eigvalS[idx]
    eigvecS = eigvecS[:,idx]

    #V MATRIX CONSTRUCTION and TRANSPOSE

    V = np.empty([4,4], dtype = float)

    for i in range (0, 4):
        V[:,i] = (1 / m.sqrt(eigvalS[i]) ) * eigvecS[:,i]

    Vtr = V.transpose()
    Fpr = np.matmul(Vtr, F)
    Fpr = np.matmul(Fpr, V)

    eigvalF, eigvecF = la.eig(F)

    eigvalF = eigvalF.real
    eigvecF = eigvecF.real

    idx = eigvalF.argsort()[::1]
    eigvalF = eigvalF[idx]
    eigvecF = eigvecF[:,idx]

    Cnew = np.matmul(V, eigvecF)

    return eigvalF, Cnew


# @njit
def iteration(H, P, S):

    i = 0

    Eoldt = -100
    Enewt = -1

    while (abs(Enewt - Eoldt) > delta):

        if i == 0 :

            C = np.zeros((4,4))
            C_old = C

        Eoldt = Enewt

        F = fock(H, P, C, C_old)

        Enew, Cnew = diagonalisation(F, S)

        C_old = C

        C = Cnew

        Enewt = Enew[0]

        print(i)

        i += 1


    return Enew, Cnew, F, C_old


# Compute matrices with dedicated function

H, P, S = integrals(alphas)

# Set initial values for the vector of the coefficients

Enew, Cnew, F, C_old = iteration(H, P, S)
