#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:08:29 2022

@author: siriapasini
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import eigh

Z = 4 # For Berillium

alfa = np.array([0.7064859542E+02, 0.1292782254E+02, 0.3591490662E+01, 0.1191983464E+01, 0.3072833610E+01, 0.6652025433E+00, 0.2162825386E+00, 0.8306680972E-01])


epsilon = 0.00001
beta = 0.01

def H(alfa):
    
    # SINGLE PARTICLE HAMILTONIAN

    H_mat = np.zeros((8,8))

    for i in range(0,8,1):

        j = 0

        while j <= i:

            H_mat[i,j] =  3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]
                            + alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j])

            H_mat[j,i] = H_mat[i,j]

            j += 1
    return H_mat

    # OVERLAP MATRIX
def S(alfa):

    S = np.zeros((8,8))

    for i in range(0,8,1):

        j = 0

        while j <= i:

            S[i,j] =  (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)

            S[j,i] = S[i,j]

            j += 1


    return S

def nor(c,alfa):
    norm = 0
    cnew = np.zeros((8,1))
    for i in range(0,8):
        for j in range(0,8):
            norm += c[i]*c[j]* (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)
            
    for i in range(0,8):
        cnew[i] = c[i] / norm**(1/2)
        
    return cnew 

#we are directly calculating the exchange and direct term

def F(c,alfa):
    F_mat = np.zeros((8,8))
    for i in range (0,8):
       for j in range (0,8):
           for k in range(0,8):
                for l in range(0,8):
                    for e in range(0,2):
                        F_mat[i,j] =  F_mat[i,j] + c[k,e]*c[l,e]* 2*m.pi**(5/2)/(alfa[k]+alfa[l]+alfa[i]+alfa[j])**(1/2) * ( 2/ ((alfa[i]+alfa[j]) * (alfa[k]+alfa[l])) -1/((alfa[i]+alfa[k])*(alfa[j]+alfa[l]))) 
    return F_mat
    
  
# # step zero 
# Hmat= H(alfa)
# S = S(alfa)
# eigvals, eigvecs = eigh(Hmat, S, eigvals_only=False)

# Enew = eigvals[0]
    
# c = abs((eigvecs[:,0]))
Eold= np.zeros((2,1))
Enew= np.ones((2,1))*10
Hmat= H(alfa)
S = S(alfa)
c = np.zeros((8,2))


#self consistent solution
while (abs(Enew[0]-Eold[0]) > epsilon) or (abs(Enew[1]-Eold[1]) > epsilon) :
        Eold = Enew
        Fmat = F(c,alfa)
        eigvals, eigvecs = eigh(Fmat+Hmat, S, eigvals_only=False)
        Enew = np.array([eigvals[0],eigvals[1]])
        cnew = np.column_stack((eigvecs[:,0], eigvecs[:,1]))
        for i in range (0,8):
            for j in range (0,2):
                c[i,j] = cnew[i,j] * beta + (1-beta) * c[i,j]
        

cfinal1 = nor(c[:,0],alfa)
cfinal2 = nor(c[:,1],alfa)

        
#compute the total energy
a = 0
b = 0
for i in range(0,8):
    for j in range(0,8):
            a += (cfinal1[i]*cfinal1[j]+ cfinal2[i]*cfinal2[j])*Hmat[i,j]
            b += (cfinal1[i]*cfinal1[j]+ cfinal2[i]*cfinal2[j])*Fmat[i,j]
        
Etot1 = 2*a+b 
Etot2 = 2*(Enew[0]+Enew[1])-b 

# now we plot
x = np.linspace(0,6,200)
f1s = cfinal1[0]* np.exp(-alfa[0]*x**2) +cfinal1[1]* np.exp(-alfa[1]*x**2) +cfinal1[2]* np.exp(-alfa[2]*x**2) + cfinal1[3]* np.exp(-alfa[3]*x**2)+cfinal1[4]* np.exp(-alfa[4]*x**2)+cfinal1[5]* np.exp(-alfa[5]*x**2) + cfinal1[6]* np.exp(-alfa[6]*x**2)+cfinal1[7]* np.exp(-alfa[7]*x**2)
f2s = cfinal2[0]* np.exp(-alfa[0]*x**2) +cfinal2[1]* np.exp(-alfa[1]*x**2) +cfinal2[2]* np.exp(-alfa[2]*x**2) + cfinal2[3]* np.exp(-alfa[3]*x**2)+cfinal2[4]* np.exp(-alfa[4]*x**2)+cfinal2[5]* np.exp(-alfa[5]*x**2) + cfinal2[6]* np.exp(-alfa[6]*x**2)+cfinal2[7]* np.exp(-alfa[7]*x**2)
# plot the function
plt.figure()
plt.plot(x,f1s,'r',color='green', label="1s orbital")
plt.plot(x,f2s,'r',label= "2s orbital")
plt.legend()
plt.grid()
plt.ylabel('$\phi(r)$')
plt.xlabel('r')
plt.title("Berillium 1s and 2s orbital")
plt.savefig("berillium.png", dpi=300)


    