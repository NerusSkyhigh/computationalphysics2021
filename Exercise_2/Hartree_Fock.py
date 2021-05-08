
import math as m
import numpy as np
#import matplotlib.pyplot as plt 
from numba import njit
import scipy.linalg as la

Z = 2 # For Helium

alfa = np.array([13.00773, 1.962079, 0.444529, 0.1219492]) # DA CAMBIAREEEEEEEEE

a = 0.001
delta = 0.001


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
            
            
    # 2-BODY POTENTIAL
            
    Pot_mat = np.zeros((16,16))

    for p in range(0, 4, 1):   
        
        for q in range(0, p+1, 1):
            
            for r in range(0, 4, 1):
                
                for s in range(0, r+1, 1):
                    
                    if (4*s+r <= 4*q+p):
                        
                        Pot_mat[4*q+p, 4*s+r] = m.pi**(5.0/2.0) / (4 * alfa[p] 
                        * alfa[q] * alfa[r] * alfa[s]) * (m.sqrt(alfa[p] + alfa[q]) 
                        * m.sqrt(alfa[r] + alfa[s]) / m.sqrt(alfa[p] + alfa[q] + alfa[r]
                        + alfa[s])) * S[p,q] * S[r,s]
                                                
    # Now we must assign a value to the missing elements in the "upper part"
          
    for i in range(0,16,1):
        
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
        
            
    return H_mat, Pot_mat, S


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

H, P, S = integrals(alfa) 
    
# Set initial values for the vector of the coefficients 

Enew, Cnew, F, C_old = iteration(H, P, S)
    

        