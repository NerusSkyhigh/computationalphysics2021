#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:36:03 2022

@author: siriapasini
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import eigh

Z = 2 # For Helium

alfa = np.array( [14.899983, 2.726485, 0.757447, 0.251390] ) 

epsilon = 0.00001
beta = 0.01

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
        cnew[i] = abs(c[i] / norm**(1/2))
        
    return cnew 

#we are directly calculating the exchange and direct term

def F(c,alfa):
    F_mat = np.zeros((4,4))
    for i in range (0,4):
       for j in range (0,4):
           for k in range(0,4):
                for l in range(0,4):
                    F_mat[i,j] =  F_mat[i,j] + c[k]*c[l]* 2*m.pi**(5/2)/(alfa[k]+alfa[l]+alfa[i]+alfa[j])**(1/2) * ( 2/ ((alfa[i]+alfa[j]) * (alfa[k]+alfa[l])) -1/((alfa[i]+alfa[k])*(alfa[j]+alfa[l]))) 
    return F_mat
    
  
# # step zero 
# Hmat= H(alfa)
# S = S(alfa)
# eigvals, eigvecs = eigh(Hmat, S, eigvals_only=False)

# Enew = eigvals[0]
    
# c = abs((eigvecs[:,0]))
Eold = 0
Enew= 10
c = np.zeros((4,1))
Hmat= H(alfa)
S = S(alfa)


#self consistent solution
while abs(Enew-Eold) > epsilon:
    Eold = Enew
    Fmat = F(c,alfa)
    eigvals, eigvecs = eigh(Fmat+Hmat, S, eigvals_only=False)
    Enew = eigvals[0]
    cnew = eigvecs[:,0]
    for i in range (0,4):
        c[i] = cnew[i] * beta + (1-beta) * c[i]

cfinal = nor(c,alfa)
        
#compute the total energy
a = 0
b = 0
for i in range(0,4):
    for j in range(0,4):
        a += cfinal[i]*cfinal[j]*Hmat[i,j] 
        b += cfinal[i]*cfinal[j]*Fmat[i,j]
        
Etot1 = 2*a+b 
Etot2 = 2*Enew-b 
# now we plot
x = np.linspace(0,6,200)
f = cfinal[0]* np.exp(-alfa[0]*x**2) +cfinal[1]* np.exp(-alfa[1]*x**2) +cfinal[2]* np.exp(-alfa[2]*x**2) + cfinal[3]* np.exp(-alfa[3]*x**2)
freal = ((1.7**3/2)/(m.pi)**(1/2))  * np.exp(-1.7*x)
density = 2*(abs(f)**2+abs(f)**2)*x**2*4*m.pi
# plot the function
plt.plot(x,f,'r',color='green', label="Numerical function")
plt.plot(x,freal,'r',label= "Exact function")
plt.legend()
plt.grid()
plt.ylabel('$\phi(r)$')
plt.xlabel('r')
plt.title("Helium orbital")
plt.savefig("orbital.png", dpi=300)
    
plt.figure()
plt.plot(x,density,'r',label= "1s")
plt.legend()
plt.grid()
plt.ylabel('$4\pi r^{2} \\rho(r)$')
plt.xlabel('r')
plt.title("Helium electronic radial distribution")
plt.savefig("He_density.png", dpi=300)
 
    
    
    
    
    
    
    
    