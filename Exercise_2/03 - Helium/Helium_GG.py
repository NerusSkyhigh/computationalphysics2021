import math as m
import numpy as np
from numba import njit

Z = 2 # For Helium

ALFA = np.array([14.899983, 2.726485, 0.757447, 0.251390] )

MIX_COEFF = 0.01
DELTA = 0.0001


# Computes the single particle, direct and exchange integrals with given orbitals
@njit
def computeHamiltonian(alfa):
    # SINGLE PARTICLE HAMILTONIAN
    H = np.zeros((4,4))
    for i in range(0, 4):
        for j in range(0, i+1):
            H[i,j] = H[j,i] = (3*(alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / \
                              (( alfa[i]+ alfa[j])**(5.0/2.0) ) \
                              - Z * 2 * m.pi / (alfa[i] + alfa[j]) )
    return H

@njit
def computeOverlapMatrix(alfa):
    # OVERLAP MATRIX
    S = np.zeros((4,4))
    for i in range(0, 4):
        for j in range(0, i+1):
            S[i,j] = S[j,i] = (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

    return S

@njit
def computeTwoBodyPotential(alfa):
    # 2-BODY POTENTIAL
    Pot_mat = np.zeros((16,16))
    for p in range(0, 4):
        for q in range(0, p+1):
            for r in range(0, 4):
                for s in range(0, r+1):
                    if (4*s+r <= 4*q+p):
                        Pot_mat[4*q+p, 4*s+r] = (2 * m.pi**(5.0/2.0)) / \
                        ((alfa[p] + alfa[q]) * (alfa[r] + alfa[s]) \
                          * m.sqrt(alfa[p] + alfa[q] + alfa[r] + alfa[s] ))

    # Now we must assign a value to the missing elements in the "upper part"
    for i in range(0, 16):
        for j in range(0, i):
            Pot_mat[j,i] = Pot_mat[i,j]

    # Now we fill the missing rows and columns
    for q in range (0, 4):
        for p in range (q, 4):
            Pot_mat[(4*p +q), :] = Pot_mat[(4*q + p), :]


    for q in range (0,4):
        for p in range (q, 4):
            Pot_mat[:, (4*p +q)] = Pot_mat[:, (4*q + p)]

    return Pot_mat


@njit
def diagonalisation(F, V):
    # Transformed Fock matrix
    Ftr = np.dot( np.dot( np.transpose(V), F), V)

    eigvalF, eigvecF = np.linalg.eigh(Ftr)

    # Eigenvectors of the general eigenvalue problem
    C = np.dot(V, eigvecF)

    return eigvalF, C

@njit
def computeDensityMatrix(C):
    P = np.zeros(((4, 4)))

    for r in range(0, 4):
        for s in range(0, 4):
            # Sum over the number of electrons. Alternatively sum over half
            # the number of electrons and add a factor 2 in front of the product
            for k in range(0, 1):
                P[r,s] += C[r,k] * C[s,k]

    return P


@njit
def computeFockOperator(H, P, Vtb):
    F = H.copy()

    for i in range(0, 4):
        for j in range(0, 4):

            for k in range(0, 4):
                for l in range(0, 4):

                    F[i,j] += P[k,l] * (2.0 * Vtb[4*j +i, 4*l + k] \
                                            - Vtb[4*l +i, 4*j + k] )

    return F


def computeEnergy(P, H, F):
    E = 0

    for r in range(0, 4):
        for s in range(0, 4):
            E += P[r,s] * (H[r,s] + F[r,s])

    return E



def iteration(H, Vtb, V):
    i = 0
    Egs = -1*np.inf # Ground state Energy. What we look for
    diff = 1

    P = np.zeros((4,4))

    while diff > DELTA:

        P_old = P.copy()

        # Update the density matrix with the Mixing procedure
        P = MIX_COEFF * P + (1 - MIX_COEFF) * P_old

        F = computeFockOperator(H, P, Vtb)

        # C is the value of the coefficients of the base
        # Eigval the eigenvalues vector
        Eigval, C = diagonalisation(F, V)

        # Calculate the density matrix from the coefficients
        P = computeDensityMatrix(C)

        Egs = computeEnergy(P, H, F)

        # The difference between the old density matrix and the current
        # is evaluated as the norm of the difference of the two matrices.
        diff = np.linalg.norm(P-P_old)

        print("Step: " +str(i) + ", Energy: " + str(Egs) )
        i += 1

    return Egs, C, F, P, Eigval



# ------------ MAIN ------------
# Calculate integrals
H = computeHamiltonian(ALFA)
Vtb = computeTwoBodyPotential(ALFA)
S = computeOverlapMatrix(ALFA)


# Diagonalise S matrix
eigvalS, eigvecS = np.linalg.eigh(S)

# Construct diagonal eigenvalues matrix
Seighlf = np.diag( eigvalS**(-0.5) )

# Compute the V matrix from the eigenvectors of S
# We exploit the fact that Seighlf is diagonal
V = np.dot(eigvecS, Seighlf)


# Main cycle of computation
Egs, C, F, P, Eigvals = iteration(H, Vtb, V)



# Print results
print("\nThe energy is {} hartrees".format(Egs) )
