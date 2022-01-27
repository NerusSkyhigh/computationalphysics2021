#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:42:43 2022

@author: siriapasini
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import eigh

Z = 1 # For Hydrogen

alfa = np.array([13.00773, 1.962079, 0.444529, 0.1219492])

# Computes the single particle, direct and exchange integrals with given orbitals

def H(alfa):
    
    # SINGLE PARTICLE HAMILTONIAN

    H_mat = np.zeros((4,4))

    for i in range(0,4,1):

        j = 0

        while j <= i:

            H_mat[i,j] =  3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]
                            + alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j])

            H_mat[j,i] = H_mat[i,j]

            j += 1
    return H_mat

    # OVERLAP MATRIX
def S(alfa):

    S = np.zeros((4,4))

    for i in range(0,4,1):

        j = 0

        while j <= i:

            S[i,j] =  (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

            S[j,i] = S[i,j]

            j += 1


    return S

def nor(c,alfa):
    norm = 0
    cnew = np.zeros((4,1))
    for i in range(0,4):
        for j in range(0,4):
            norm += c[i]*c[j]* (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)
            
    for i in range(0,4):
        cnew[i] = c[i] / norm**(1/2)
        
    return cnew    
        




H= H(alfa)
S = S(alfa)

eigvals, eigvecs = eigh(H, S, eigvals_only=False)

E = eigvals[0]
    
c = abs((eigvecs[:,0]))

#now we want to normalize the coefficients
cnew = nor(c,alfa)

# now we plot
x = np.linspace(0,6,200)
f = cnew[0]* np.exp(-alfa[0]*x**2) +cnew[1]* np.exp(-alfa[1]*x**2) +cnew[2]* np.exp(-alfa[2]*x**2) + cnew[3]* np.exp(-alfa[3]*x**2)
freal = (1/(m.pi)**(1/2))  * np.exp(-x)
density = abs(f)**2*x**2*4*m.pi
# plot the function
plt.figure()
plt.plot(x,f,'r',color='green', label="Numerical function")
plt.plot(x,freal,'r',label= "Exact function")
plt.legend()
plt.grid()
plt.ylabel('$\psi(r)$')
plt.xlabel('r')
plt.title("Hydrogen orbital")
plt.savefig("orbital.png", dpi=300)

plt.figure()
plt.plot(x,density,'r',label= "1s")
plt.legend()
plt.grid()
plt.ylabel('$4 \pi r^{2} \\rho(r)$')
plt.xlabel('r')
plt.title("Hydrogen electronic  radial distribution")
plt.savefig("H_density.png", dpi=300)


