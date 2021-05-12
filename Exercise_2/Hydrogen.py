import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

Z = 1 # For Hydrogen

alfa = np.array([13.00773, 1.962079, 0.444529, 0.1219492])

# Computes the single particle, direct and exchange integrals with given orbitals

def integrals(alfa):
    
    # SINGLE PARTICLE HAMILTONIAN

    H_mat = np.zeros((4,4))

    for i in range(0,4,1):

        j = 0

        while j <= i:

            H_mat[i,j] =  3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]
                            + alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j])

            H_mat[j,i] = H_mat[i,j]

            j += 1

    # OVERLAP MATRIX

    S = np.zeros((4,4))

    for i in range(0,4,1):

        j = 0

        while j <= i:

            S[i,j] =  (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

            S[j,i] = S[i,j]

            j += 1


    return H_mat, S


# Function for solving the generalised eigenvalue problem

def diagonalisation(H, S):

    # Diagonalise S matrix

    eigvalS, eigvecS = np.linalg.eigh(S)

    Seighlf = ( np.diag(eigvalS**(-0.5) ) )

    V = np.dot(eigvecS, Seighlf)
    
    Hpr = np.dot(np.dot(np.transpose(V), H), V)

    eigvalH, eigvecH = np.linalg.eig(Hpr)

    idx = eigvalS.argsort()[::1]
    eigvalH = eigvalH[idx]
    eigvecH = eigvecH[:,idx]

    eigvecH = np.dot(V, eigvecH)

    return eigvalH, eigvecH


# CALCULATION

H, S = integrals(alfa)

eigval, eigvec = diagonalisation(H,S)

eigvec = eigvec[:,3]

# Define the integrand function for normalisation

def G(x, a, b):
    return a * m.exp(-b * x**2)

def GT2(x, eigenvec, alfa):
    return (G(x, eigvec[0], alfa[0]) + G(x, eigvec[1], alfa[1]) \
            + G(x, eigvec[2], alfa[2]) + G(x, eigvec[3], alfa[3]))**2
        
def Gex2(x):
    return (1 / m.sqrt(m.pi) * m.exp(-1 * x))**2

# Verify this is normalised

I = quad(GT2, 0, np.inf, args=(eigvec,alfa))
I2 = quad(Gex2, 0, np.inf)

print("The integral of the square modulus is {} pm {} ".format(I[0],I[1]))
print("The integral of the square modulus of the exact function is {} pm {} ".format(I2[0],I2[1]))

# Functions Drawing

def Gdr(x, a, b):
    return a * np.exp(-b * np.power(x,2))  / m.sqrt(I[0])

def GT(x, eigenvec, alfa):
    return ( Gdr(x, eigvec[0], alfa[0]) + Gdr(x, eigvec[1], alfa[1]) \
        + Gdr(x, eigvec[2], alfa[2]) + Gdr(x, eigvec[3], alfa[3]) )

def Gex(x):
    return ( 1 / m.sqrt(m.pi) ) * np.exp(-1 * x) / m.sqrt(I2[0])

def GT2(x, eigenvec, alfa):
    return (Gdr(x, eigvec[0], alfa[0]) + Gdr(x, eigvec[1], alfa[1]) \
        + Gdr(x, eigvec[2], alfa[2]) + Gdr(x, eigvec[3], alfa[3]) )**2
        
def Dex(x):
    return np.power(1 / m.sqrt(m.pi) * np.exp(-1 * x) , 2) / I2[0]

r = np.arange(0,5,0.01)

# Plot the gaussian, the total wavefunction and the exact result 

fig = plt.figure()
plt.plot(r, -Gdr(r, eigvec[0], alfa[0]), label="Gaussian 1")
plt.plot(r, -Gdr(r, eigvec[1], alfa[1]), label = "Gaussian 2")
plt.plot(r, -Gdr(r, eigvec[2], alfa[2]), label = "Gaussian 3")
plt.plot(r, -Gdr(r, eigvec[3], alfa[3]), label = "Gaussian 4")
plt.plot(r, -GT(r, eigvec, alfa), label = "GTO Orbital")
plt.plot(r, Gex(r), label = "Exact result")
plt.legend()
plt.grid()
plt.title("Orbital")
plt.savefig("Gaussians.png", dpi=300)

# Electronic density plot 

fig = plt.figure()
plt.plot(r, GT2(r, eigvec, alfa), label="Density")
plt.plot(r, Dex(r), label="Exact Density")
plt.legend()
plt.grid()
plt.title("Hydrogen Electronic Density")
plt.savefig("Density_H.png", dpi=300)
