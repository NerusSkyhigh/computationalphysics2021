import math as mt
import numpy as np
import time
from scipy.integrate import quad
import matplotlib.pyplot as plt

Z = 4 # For Berillium

# Coefficients for the exponents of the gaussians

alfa = np.array([70.64859542, 12.92782254, 3.591490662, 1.191983464, 
                 3.072833610, 0.6652025433, 0.2162825386, 0.08306680972] )


a = 0.025 # Mixing parameter
delta = 1E-10 # Convergence threshold
dim = 8 # number of basis functions
Nelec = 4 # Number of electrons

# Computes the single particle, direct and exchange integrals with given orbitals

def integrals(alfa):

    # SINGLE PARTICLE HAMILTONIAN  (one electron only)

    H_mat = np.zeros((dim,dim))

    for i in range(0,dim,1):

        j = 0

        while j <= i:

            H_mat[i,j] = ( 3 * (alfa[i] * alfa[j] * mt.pi**(3.0/2.0) ) / (( alfa[i]
                            + alfa[j])**(5.0/2.0) ) - Z*2*mt.pi/(alfa[i]+alfa[j]))

            H_mat[j,i] = H_mat[i,j]

            j += 1


    # OVERLAP MATRIX

    S = np.zeros((dim,dim))

    for i in range(0,dim,1):

        j = 0

        while j <= i:

            S[i,j] =  (mt.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

            S[j,i] = S[i,j]

            j += 1


    # 2-BODY POTENTIAL

    Exch_mat = np.zeros((int(dim**2), int(dim**2)))

    for p in range(0, dim, 1):

        for q in range(0, p+1, 1):

            for r in range(0, dim, 1):

                for s in range(0, r+1, 1):

                    if (dim*s+r <= dim*q+p):

                        Exch_mat[dim*q+p, dim*s+r] = (2 * mt.pi**(5.0/2.0)) / \
                        ((alfa[p] + alfa[q]) * (alfa[r] + alfa[s]) \
                          * mt.sqrt(alfa[p] + alfa[q] + alfa[r] + alfa[s] ))


    # Now we must assign a value to the missing elements in the "upper part"

    for i in range(0,int(dim**2),1):

        j = 0

        while j < i:

            Exch_mat[j,i] = Exch_mat[i,j]

            j += 1

    # Now fill the missing rows and columns

    for q in range (0,dim,1):

        for p in range (q,dim,1):

            Exch_mat[(dim*p +q), :] = Exch_mat[(dim*q + p), :]

    for q in range (0,dim,1):

        for p in range (q,dim,1):

            Exch_mat[:, (dim*p +q)] = Exch_mat[:, (dim*q + p)]


    return H_mat, S, Exch_mat


# Diagonalisation procedure

def diagonalisation(F, V):

    Fpr = np.dot(np.dot(np.transpose(V), F), V)

    eigvalF, eigvecF = np.linalg.eigh(Fpr)

    Cnew = np.dot(V, eigvecF)

    return eigvalF, Cnew


# Define function to compute the density matrix

def densitymatrix(P, C):

    P_old = np.zeros((dim, dim))

    for r in range(0, dim):

        for s in range(0, dim):

            P_old[r,s] = P[r, s]
            P[r,s] = 0

            # Sum over the number of electrons. Alternatively sum over half
            # the number of electrons and add a factor 2 in front of the product

            for k in range(0, int(Nelec/2)):

                P[r,s] += C[r,k] * C[s,k]

    return P, P_old


# Define function that computes the fock operator

def fock(H, P, Exch): # Make Fock Matrix

    F = np.zeros((dim, dim))

    for i in range(0, dim):

        for j in range(0, dim):

            F[i,j] = H[i,j]

            for k in range(0, dim):

                for l in range(0, dim):

                    F[i,j] = F[i,j] + P[k,l] *(2.0* Exch[dim*j + i, dim*l + k] \
                                               - Exch[dim*l +i, dim*j + k] )

    return F


# Function to calculate the energy

def energy(P, H, F):

    EN = 0

    for r in range(0, dim):
        for s in range(0, dim):

            EN += P[r,s] * (H[r,s] + F[r,s])

    return EN


# Calculate change in density matrix using Root Mean Square Deviation (RMSD)

def deltap(D, Dold):

    DELTA = 0.0

    for i in range(0, dim):

        for j in range(0, dim):

            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA)**(0.5)

# Function for iterative process

def iteration(H, Exch, V):

    i = 0

    Enewt = -90
    diff = 1

    P_old = np.zeros((dim,dim))
    P = P_old

    while (diff > delta):

        P_mix = a * P + (1 - a) * P_old

        F = fock(H, P_mix, Exch)

        Enew, Cnew = diagonalisation(F, V)

        # Calculate the density matrix

        P, P_old = densitymatrix(P, Cnew)

        Enewt = energy(P, H, F)

        diff = deltap(P, P_old)

        print("Step: " +str(i) + ", Energy: " + str(Enewt) )

        i += 1

    return Enewt, Cnew, F, P, Enew

# Start timing

start = time.time()

# Calculate integrals

H, S ,Exch = integrals(alfa)

# Diagonalise S matrix

eigvalS, eigvecS = np.linalg.eigh(S)

# Construct diagonal eigenvalues matrix

Seighlf = ( np.diag(eigvalS**(-0.5) ) )

# V MATRIX CONSTRUCTION

V = np.dot(eigvecS, Seighlf)

# RUN CALCULATION

Enewt, Cnew, F, P, Enew = iteration(H, Exch, V)

# End timing

end = time.time()

# Print results

print("\nThe energy is {} hartrees".format(Enewt) )

print("\nThe calculation took " + str(end-start) + " seconds")

# Define the integrand function for normalisation

def G(x, a, b):
    return a * mt.exp(-b * x**2)

def GT2(x, eigvec, alfa):
    return (G(x, eigvec[0], alfa[0]) + G(x, eigvec[1], alfa[1]) \
            + G(x, eigvec[2], alfa[2]) + G(x, eigvec[3], alfa[3]) )**2

# Verify this is normalised

Cnew1s = Cnew[0:4,0]
Cnew2s = Cnew[4:9,0]
alfa1s = alfa[0:4]
alfa2s = alfa[4:9]

I1s = quad(GT2, 0, np.inf, args=(Cnew1s, alfa1s))
I2s = quad(GT2, 0, np.inf, args=(Cnew2s, alfa2s))

print("\nThe norm for the 1s wf is {} pm {} ".format(I1s[0],I1s[1]))
print("\nThe norm for the 2s wf is {} pm {} ".format(I2s[0],I2s[1]))

# Functions Drawing

def Gdr(x, a, b):
    return a * np.exp(-b * np.power(x,2)) 

def GT(x, eigvec, alfa):
    return ( Gdr(x, eigvec[0], alfa[0]) + Gdr(x, eigvec[1], alfa[1]) \
        + Gdr(x, eigvec[2], alfa[2]) + Gdr(x, eigvec[3], alfa[3]) )

def GT2(x, eigvec, alfa):
    return (Gdr(x, eigvec[0], alfa[0]) + Gdr(x, eigvec[1], alfa[1]) \
        + Gdr(x, eigvec[2], alfa[2]) + Gdr(x, eigvec[3], alfa[3]) )**2

r = np.arange(0,5,0.01)

# Plot the gaussian, the total wavefunction and the exact result

fig = plt.figure()
plt.plot(r, GT(r, Cnew1s, alfa1s) / mt.sqrt(I1s[0]), label = "GTO Orbital, 1s")
plt.plot(r, -GT(r, Cnew2s, alfa2s) / mt.sqrt(I2s[0]), label = "GTO Orbital, 2s")
plt.legend()
plt.grid()
plt.title("Orbitals")
plt.savefig("Gaussians_Berillium.png", dpi=300)

# Electronic density plot

fig = plt.figure()
plt.plot(r, GT2(r, Cnew1s, alfa1s) / I1s[0], label="Density 1s")
plt.plot(r, GT2(r, Cnew2s, alfa2s) / I2s[0], label="Density 2s")
plt.legend()
plt.grid()
plt.title("Berillium Electronic Density")
plt.savefig("Density_Be.png", dpi=300)
