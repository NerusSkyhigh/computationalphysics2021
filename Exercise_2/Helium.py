# Helium Hartree-Fock Scheme 

import math as m
import numpy as np

Z = 2 # For Helium

alfa = np.array( [14.899983, 2.726485, 0.757447, 0.251390] ) 

a = 0.01
delta = 1E-5

# Computes the single particle, direct and exchange integrals with given orbitals

def integrals(alfa):
    
    # SINGLE PARTICLE HAMILTONIAN  (one electron only)
    
    H_mat = np.zeros((4,4))
    
    for i in range(0,4,1):
        
        j = 0
        
        while j <= i:
            
            H_mat[i,j] = ( 3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]                                        
                            + alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j]) )
            
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
    
    # Here, the other algorithm obtains a  symmetric F and uses the eigenh 
    # algorithm... we encounter the same problem and our F is not symmetric,
    # probably due to the use of a different V 
    
    Fpr = np.dot(np.dot(np.transpose(V), F), V)
    
    eigvalF, eigvecF = np.linalg.eig(Fpr)
    
    eigvalF = eigvalF.real
    eigvecF = eigvecF.real
    
    idx = eigvalF.argsort()[::1]
    eigvalF = eigvalF[idx]
    eigvecF = eigvecF[:,idx]
    
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
                
                P[r,s] +=  2 * C[r,k] * C[s,k] 
                
    return P, P_old

# Define function that computes the fock operator 

def fock(H, Exch, P_mix):
    
    F = np.zeros((4, 4))
    
    for p in range(0,4,1):
        
        for q in range(0,4,1):
            
            F[p,q] = H[p,q]
            
            for r in range(0,4,1):
                
                for s in range(0,4,1):
                    
                    F[r,s] = F[r,s] + P_mix[r,s] * ( Exch[4*q + p, 4*s + r] - \
                                                    0.5 * Exch[4*s +p, 4*q +r] )
    
    return F

# Function to calculate the energy

def energy(P, H, F): # Calculate energy at iteration

    EN = 0
    
    for r in range(0, 4):
        for s in range(0, 4):
            
            EN += 0.5 * P[r,s] * (H[r,s] + F[r,s])
            
    return EN


def deltap(D, Dold): # Calculate change in density matrix using Root Mean Square Deviation (RMSD)

    DELTA = 0.0
    
    for i in range(0, 4):
        
        for j in range(0, 4):
            
            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA)**(0.5)


def iteration(H, Exch, V):
    
    i = 0
    
    Enewt = -90
    diff = 1
    
    #C = np.zeros((4,4)) 
    P_old = np.zeros((4,4))
    P = P_old
    
    while (diff > delta):
        
        P_mix = a * P + (1 - a) * P_old
        
        F = fock(H, Exch, P_mix)
        
        Enew, Cnew = diagonalisation(F, V)
        
        # Calculate the density matrix
    
        P, P_old = densitymatrix(P, Cnew)
        
        Enewt = energy(P, H, F)
        
        diff = deltap(P, P_old)
        
        print(i, Enewt, diff)
        
        i += 1 
        
    return Enewt, Cnew, F, P, P_old
    

H, Exch, S = integrals(alfa)

# Diagonalise S matrix 
# For some reason, using the algorithm for symmetric matrices does not work.

#eigvalS, eigvecS = np.linalg.eigh(S)    
eigvalS, eigvecS = np.linalg.eig(S)

eigvalS = eigvalS.real
eigvecS = eigvecS.real
    
idx = eigvalS.argsort()[::1]
eigvalS = eigvalS[idx]
eigvecS = eigvecS[:,idx]

Seighlf = (np.diag(eigvalS**(-0.5)))

# V MATRIX CONSTRUCTION
# The other algorithm provides the following V matrix. However, ours does not converge with this.

Seighlf = (np.diag(eigvalS**(-0.5)))

V = np.dot(eigvecS, np.dot(Seighlf, np.transpose(eigvecS)))

# # Our V matrix

# V = np.empty([4,4], dtype = float)

# for i in range (0, 4):
#     V[:,i] = (1 / m.sqrt(eigvalS[i]) ) * eigvecS[:,i]
    

# RUN THE CALCULATION

Enew, Cnew, F, P, P_old = iteration(H, Exch, V)
