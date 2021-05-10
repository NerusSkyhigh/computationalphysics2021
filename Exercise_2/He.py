#import sys
import numpy as np
import math as m
import matplotlib.pyplot as plt

Z = 2 # For Helium

alfa = np.array( [14.899983, 2.726485, 0.757447, 0.251390] ) # Values of exponent coefficients

dim = 4 

Nelec = 2

def matrices(alfa):
    
     # SINGLE PARTICLE HAMILTONIAN  (one electron)
    
    H_mat = np.zeros((dim,dim))
    
    for i in range(0,dim,1):
        
        j = 0
        
        while j <= i:
            
            H_mat[i,j] = ( 3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]                                        
                        + alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j]) )
            
            H_mat[j,i] = H_mat[i,j]
            
            j += 1
            
            
    # OVERLAP MATRIX
    
    S = np.zeros((dim,dim))
    
    for i in range(0,dim,1):
        
        j = 0
        
        while j <= i:
            
            S[i,j] =  (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)
            
            S[j,i] = S[i,j]
            
            j += 1
            
            
    # 2-BODY POTENTIAL
            
    Exch_mat = np.zeros((dim**2,dim**2))
      
    for p in range(0, dim, 1):   
        
        for q in range(0, p+1, 1):
            
            for r in range(0, dim, 1):
                
                for s in range(0, r+1, 1):
                    
                    if (4*s+r <= 4*q+p):
                        
                        Exch_mat[4*q+p, 4*s+r] = (2 * m.pi**(5.0/2.0)) / \
                        ((alfa[p] + alfa[q]) * (alfa[r] + alfa[s]) \
                          * m.sqrt(alfa[p] + alfa[q] + alfa[r] + alfa[s] ))
                                                      
                                                      
    # Now we must assign a value to the missing elements in the "upper part"
          
    for i in range(0,dim**2,1):
        
        j = 0
        
        while j < i:
            
            Exch_mat[j,i] = Exch_mat[i,j]
            
            j += 1                                                      
                                                          
    # Now fill the missing rows and columns

    for q in range (0,dim,1):
        
        for p in range (q,dim,1):
            
            Exch_mat[(4*p +q), :] = Exch_mat[(4*q + p), :]
            
    for q in range (0,dim,1):
        
        for p in range (q,dim,1):
            
            Exch_mat[:, (4*p +q)] = Exch_mat[:, (4*q + p)]
            
            
    return H_mat, Exch_mat, S



def fprime(X, F): # Put Fock matrix in orthonormal AO basis
    return np.dot(np.transpose(X), np.dot(F, X)) 

def makedensity(C, D, dim, Nelec): # Make density matrix and store old one to test for convergence

    Dold = np.zeros((dim, dim))
    
    for mu in range(0, dim):
        
        for nu in range(0, dim):
            
            Dold[mu,nu] = D[mu, nu]
            
            D[mu,nu] = 0
            
            for n in range(0, int(Nelec/2)):
                
                D[mu,nu] = D[mu,nu] + 2*C[mu,n]*C[nu,n]

    return D, Dold 

def makefock(H, P, Exch, dim): # Make Fock Matrix

    F = np.zeros((dim, dim))
    
    for i in range(0, dim):
        
        for j in range(0, dim):
            
            F[i,j] = H[i,j]
            
            for k in range(0, dim):
                
                for l in range(0, dim):
                    
                    F[i,j] = F[i,j] + P[k,l] *(Exch[4*j + i, 4*l + k] - 0.5 * Exch[4*l +i, 4*j + k] )
    
    return F 

def deltap(D, Dold): # Calculate change in density matrix using Root Mean Square Deviation (RMSD)

    DELTA = 0.0
    
    for i in range(0, dim):
        
        for j in range(0, dim):
            
            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA)**(0.5)

def currentenergy(D, Hcore, F, dim): # Calculate energy at iteration

    EN = 0
    
    for mu in range(0, dim):
        
        for nu in range(0, dim):
            
            EN += 0.5 * D[mu,nu]*(Hcore[mu,nu] + F[mu,nu])
            
    return EN


H, Exch, S = matrices(alfa)

SVAL, SVEC   = np.linalg.eigh(S) # Diagonalize basis using symmetric orthogonalization 

SVAL_minhalf = (np.diag(SVAL**(-0.5))) # Inverse square root of eigenvalues

S_minhalf    = np.dot(SVEC, np.dot(SVAL_minhalf, np.transpose(SVEC)))

P            = np.zeros((dim, dim)) # P represents the density matrix, Initially set to zero.
OLDP         = np.zeros((dim, dim)) # P represents the density matrix, Initially set to zero.

DELTA        = 1 # Set placeholder value for delta

count        = 0 # Count how many SCF cycles are done, N(SCF)

a = 0.5


while DELTA > 0.00001:
    count     += 1                             # Add one to number of SCF cycles counter
    
    P_mix     = a * P + (1-a) * OLDP
    
    F         = makefock(H, P, Exch, dim)        # Calculate Fock matrix, F
    
    Fprime    = fprime(S_minhalf, F)           # Calculate transformed Fock matrix, F'
    
    E, Cprime = np.linalg.eigh(Fprime)         # Diagonalize F' matrix
    
    C         = np.dot(S_minhalf, Cprime)      # 'Back transform' the coefficients into original basis using transformation matrix
    
    P, OLDP   = makedensity(C, P, dim, Nelec)  # Make density matrix
    
    DELTA     = deltap(P, OLDP)                # Test for convergence. If criteria is met exit loop and calculate properties of interest
    
    print("E = {:.6f}, N(SCF) = {}".format(currentenergy(P, H, F, dim), count))

print("SCF procedure complete, TOTAL E(SCF) = {} hartrees".format(currentenergy(P, H, F, dim)))

