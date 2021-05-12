import math as m
import numpy as np
#import matplotlib.pyplot as plt
from numba import njit
import scipy.linalg as la

Z = 2 # For Helium

ALFA = np.array([14.899983, 2.726485, 0.757447, 0.251390] )

MIX_COEFF = 0.01
DELTA = 0.0001


# Computes the single particle, direct and exchange integrals with given orbitals
@njit
def integrals(alfa):
    # SINGLE PARTICLE HAMILTONIAN
    H_mat = np.zeros((4,4))
    for i in range(0, 4):
        for j in range(0, i+1):
            H_mat[i,j] = H_mat[j,i] =( 3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / \
                                       (( alfa[i]+ alfa[j])**(5.0/2.0) ) \
                                     - Z * 2 * m.pi / (alfa[i] + alfa[j]) )

    # OVERLAP MATRIX
    S = np.zeros((4,4))
    for i in range(0, 4):
        for j in range(0, i+1):
            S[i,j] = S[j,i] = (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

    # 2-BODY POTENTIAL
    Pot_mat = np.zeros((16,16))
    for p in range(0, 4):
        for q in range(0, p+1):
            for r in range(0, 4):
                for s in range(0, r+1):
                    if (4*s+r <= 4*q+p):
                        # Check pdf "integrals.pdf"
                        Pot_mat[4*q+p, 4*s+r] = 2/m.sqrt(m.pi) * \
                            (m.sqrt(alfa[p] + alfa[q]) * m.sqrt(alfa[r] + alfa[s]) / \
                              m.sqrt(alfa[p] + alfa[q] + alfa[r] + alfa[s])) \
                              * S[p,q] * S[r,s]


    # Now we must assign a value to the missing elements in the "upper part"
    for i in range(0, 16):
        for j in range(0, i):
            Pot_mat[j,i] = Pot_mat[i,j]

    # Now fill the missing rows and columns
    for q in range (0, 4):
        for p in range (q, 4):
            Pot_mat[(4*p +q), :] = Pot_mat[(4*q + p), :]


    for q in range (0,4):
        for p in range (q, 4):
            Pot_mat[:, (4*p +q)] = Pot_mat[:, (4*q + p)]

    return H_mat, Pot_mat, S



# Building the Fock Matrix, representing the Fock operator
def fock(H, P, C, C_old):
    F = np.zeros((4,4))
    G = np.zeros((4,4))

    DensMat = 2*C @ C.transpose()
    DensMat_old = 2*C_old @ C_old.transpose()

    C = MIX_COEFF * DensMat + (1-MIX_COEFF)*DensMat_old


    for p in range(0, 4):
        for q in range(0, 4):
            #twob = 0.0

            for r in range(0, 4):
                for s in range(0, 4):
                    P_term = P[4*q + p, 4*s + r] - 0.5*P[4*s +p, 4*q +r]

                    #twob +=  C[r, s] * P_term
                    G[p, q] += C[r, s] * P_term

            F[p,q] = H[p,q] + G[p, q] #+ twob

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



def iteration(H, P, S):
    i = 0

    Eoldt = -100
    Enewt = -90
    diff = 100

    C = np.zeros([4, 1])
    C_old = C

    while (diff > DELTA):
        Eoldt = Enewt

        F = fock(H, P, C, C_old)

        Enew, Cnew = diagonalisation(F, S)

        C_old = C
        #C = np.column_stack(( Cnew[:, 0], Cnew[:, 0]))
        C = np.array([Cnew[:, 0]]).T

        Enewt = Enew[0]

        diff = abs(Enewt - Eoldt)

        print(i, Enewt)

        i += 1
        if i == 10000:
            break


    return Enew, Cnew, F, C_old, C



# Compute matrices with dedicated function
H, P, S = integrals(ALFA)
Enew, Cnew, F, C_old, C = iteration(H, P, S)
