import math as m
import numpy as np
import matplotlib.pyplot as plt 
from numba import njit
import scipy.linalg as la
from scipy.integrate import quad

Z = 1 # For Hydrogen

alfa = np.array([13.00773, 1.962079, 0.444529, 0.1219492])

# Computes the single particle, direct and exchange integrals with given orbitals

@njit
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
    Hpr = np.matmul(Vtr, H)
    Hpr = np.matmul(Hpr, V)
    
    eigvalH, eigvecH = la.eig(Hpr)
    
    eigvalH = eigvalH.real
    eigvecH = eigvecH.real
    
    idx = eigvalS.argsort()[::1]
    eigvalH = eigvalH[idx]
    eigvecH = eigvecH[:,idx]
    
    eigvecH = np.matmul(V, eigvecH)
    
    return eigvalH, eigvecH
    

# Calculation

H, S = integrals(alfa)    

eigval, eigvec = diagonalisation(H,S)

eigvec = eigvec[:,3]

# Define the integrand function for normalisation

def G(x, a, b):
    return a * m.exp(-b * x**2)


def GT2(x, eigenvec, alfa):
    return 4 * m.pi * (x**2) * (G(x, eigvec[0], alfa[0]) + G(x, eigvec[1], alfa[1]) + 
            G(x, eigvec[2], alfa[2]) + G(x, eigvec[3], alfa[3]))**2

# Verify this little shit is normalised

I = quad(GT2, 0, np.inf, args=(eigvec,alfa))

print("The integral of the square modulus is {} pm {} ".format(I[0],I[1]))

# Functions Drawing 

def Gdr(x, a, b):
    return a * np.exp(-b * np.power(x,2))

def GT(x, eigenvec, alfa):
    return Gdr(x, eigvec[0], alfa[0]) + Gdr(x, eigvec[1], alfa[1]) + Gdr(x, eigvec[2], alfa[2]) + Gdr(x, eigvec[3], alfa[3])

def Gex(x):
    return 1 / m.sqrt(m.pi) * np.exp(-1 * x)
    
    
r = np.arange(0,5,0.1)

G1 = Gdr(r, eigvec[0], alfa[0])
G2 = Gdr(r, eigvec[1], alfa[1])
G3 = Gdr(r, eigvec[2], alfa[2])
G4 = Gdr(r, eigvec[3], alfa[3])
G_all = GT(r, eigvec, alfa)
G_ex = Gex(r)

fig = plt.figure()
plt.plot(r, G1, label="Gaussian 1")
plt.plot(r, G2, label = "Gaussian 2")
plt.plot(r, G3, label = "Gaussian 3")
plt.plot(r, G4, label = "Gaussian 4")
plt.plot(r, G_all, label = "GTO Orbital")
plt.plot(r, G_ex, label = "Exact result")
plt.legend()
plt.title("Four Gaussians")
plt.savefig("Gaussians.png", dpi=300)







