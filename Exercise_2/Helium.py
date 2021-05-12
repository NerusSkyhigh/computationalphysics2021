# Helium Hartree-Fock Scheme 

import math as m
import numpy as np
import time

Z = 2 # For Helium

alfa = np.array([14.899983, 2.726485, 0.757447, 0.251390]) 

a = 0.001
delta = 1E-8

# Computes the single particle, direct and exchange integrals with given orbitals

def integrals(alfa):
    
    # SINGLE PARTICLE HAMILTONIAN  (one electron only)
    
    H_mat = np.zeros((4,4))
    
    for i in range(0,4,1):
        
        j = 0
        
        while j <= i:
            
            H_mat[i,j] = ( 3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]                                        
                            + alfa[j])**(5.0/2.0) ) - Z*2*m.pi/(alfa[i]+alfa[j]))
            
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
            
    Exch_mat = np.zeros((16,16))
      
    for p in range(0, 4, 1):   
        
        for q in range(0, p+1, 1):
            
            for r in range(0, 4, 1):
                
                for s in range(0, r+1, 1):
                    
                    if (4*s+r <= 4*q+p):
                        
                        Exch_mat[4*q+p, 4*s+r] = (2 * m.pi**(5.0/2.0)) / \
                        ((alfa[p] + alfa[q]) * (alfa[r] + alfa[s]) \
                          * m.sqrt(alfa[p] + alfa[q] + alfa[r] + alfa[s] ))
                                                      
                                                      
    # Now we must assign a value to the missing elements in the "upper part"
          
    for i in range(0,16,1):
        
        j = 0
        
        while j < i:
            
            Exch_mat[j,i] = Exch_mat[i,j]
            
            j += 1                                                      
                                                          
    # Now fill the missing rows and columns

    for q in range (0,4,1):
        
        for p in range (q,4,1):
            
            Exch_mat[(4*p +q), :] = Exch_mat[(4*q + p), :]
            
    for q in range (0,4,1):
        
        for p in range (q,4,1):
            
            Exch_mat[:, (4*p +q)] = Exch_mat[:, (4*q + p)]
            
            
    return H_mat, Exch_mat, S


# Diagonalisation procedure

def diagonalisation(F, V): 
    
    Fpr = np.dot(np.dot(np.transpose(V), F), V)
    
    eigvalF, eigvecF = np.linalg.eigh(Fpr)
    
    Cnew = np.dot(V, eigvecF)

    return eigvalF, Cnew


# Define function to compute the density matrix 

def densitymatrix(P, C):
    
    P_old = np.zeros((4, 4))
    
    for r in range(0, 4):
        
        for s in range(0, 4):
            
            P_old[r,s] = P[r, s]
            P[r,s] = 0
            
            # Sum over the number of electrons. Alternatively sum over half
            # the number of electrons and add a factor 2 in front of the product
            
            for k in range(0, 1): 
                
                P[r,s] += C[r,k] * C[s,k] 
                
    return P, P_old


# Define function that computes the fock operator 

def fock(H, P, Exch): # Make Fock Matrix

    F = np.zeros((4, 4))
    
    for i in range(0, 4):
        
        for j in range(0, 4):
            
            F[i,j] = H[i,j]
            
            for k in range(0, 4):
                
                for l in range(0, 4):
                    
                    F[i,j] = F[i,j] + P[k,l] *(2.0* Exch[4*j + i, 4*l + k] \
                                               - Exch[4*l +i, 4*j + k] )
    
    return F 


# Function to calculate the energy

def energy(P, H, F): 

    EN = 0
    
    for r in range(0, 4):
        for s in range(0, 4):
            
            EN += P[r,s] * (H[r,s] + F[r,s]) 
            
    return EN


# Calculate change in density matrix using Root Mean Square Deviation (RMSD)

def deltap(D, Dold): 

    DELTA = 0.0
    
    for i in range(0, 4):
        
        for j in range(0, 4):
            
            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA)**(0.5)

# Function for iterative process

def iteration(H, Exch, V):
    
    i = 0
    
    Enewt = -90
    diff = 1
    
    #C = np.zeros((4,4)) 
    P_old = np.zeros((4,4))
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

H, Exch, S = integrals(alfa)

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

# Drawing