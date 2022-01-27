#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:39:52 2022

@author: siriapasini
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import eigh

Z = 4 # For Berillium

alfa = np.array([0.7064859542E+02, 0.1292782254E+02, 0.3591490662E+01, 0.1191983464E+01, 0.3072833610E+01, 0.6652025433E+00, 0.2162825386E+00, 0.8306680972E-01])

a =  np.array([0.5675242080E-01, 0.2601413550E+00, 0.5328461143E+00, 0.2916254405E+00,-0.6220714565E-01 ,0.2976804596E-04, 0.5588549221E+00, 0.4977673218E+00])


epsilon = 0.00001
beta = 0.01

def H(alfa,a):
    
    # SINGLE PARTICLE HAMILTONIAN

    H_mat = np.zeros((2,2))

    for p in range(0,2):
        for q in range(0,2):
            for i in range(4*p,4*p+4):
                for j in range(4*q,4*q+4):
                    H_mat[p,q]+=  a[i]*a[j]* (3 * (alfa[i] * alfa[j] * m.pi**(3.0/2.0) ) / (( alfa[i]+ alfa[j])**(5.0/2.0) ) - Z * 2 * m.pi / (alfa[i] + alfa[j]))

    return H_mat

    # OVERLAP MATRIX
def S(alfa,a):

    S = np.zeros((2,2))

    for p in range(0,2):
        for q in range(0,2):
            for i in range(4*p,4*p+4):
                for j in range(4*q,4*q+4):
                    S[p,q] += a[i]*a[j] * (m.pi / (alfa[i] + alfa[j] ) )**(3.0/2.0)
    return S

def nor(c,alfa):
    norm = 0
    cnew = np.zeros((2,1))
    for i in range(0,2):
        for j in range(0,2):
            for p in range(4*i,4*i+4):
                for q in range(4*j,4*j+4):  
                    norm += c[i]*c[j]* a[p]* a[q] * (m.pi / (alfa[p] + alfa[q] ) )**(3.0/2.0)
            
    for i in range(0,2):
        cnew[i] = c[i] / norm**(1/2)
        
    return cnew 

# #we are directly calculating the exchange and direct term

def F(a,c,alfa):
    F_mat = np.zeros((2,2))
    for p in range (0,2):
        for q in range (0,2):
            for r in range(0,2):
                for s in range(0,2):
                    for n in range(4*p,4*p+4):
                        for f in range(4*q,4*q+4):
                            for l in range(4*r,4*r+4):
                                for g in range(4*s,4*s+4):
                                    for e in range(0,2):
                                        F_mat[p,q] +=  c[r,e]*c[s,e] *a[n]*a[l]*a[f]*a[g]* 2* m.pi**(5/2)/(alfa[n]+alfa[f]+alfa[l]+alfa[g])**(1/2) * ( 2/ ((alfa[n]+alfa[f]) * (alfa[l]+alfa[g])) -1/((alfa[n]+alfa[l])*(alfa[f]+alfa[g]))) 
    return F_mat
    
  
# step zero 
# Hmat= H(alfa)
# S = S(alfa)
# eigvals, eigvecs = eigh(Hmat, S, eigvals_only=False)

# Enew = eigvals[0]
    
# c = abs((eigvecs[:,0]))

Eold= np.zeros((2,1))
Enew= np.ones((2,1))*10
Hmat= H(alfa,a)
S = S(alfa,a)
c = np.zeros((2,2))



#self consistent solution
while (abs(Enew[0]-Eold[0]) > epsilon) or (abs(Enew[1]-Eold[1]) > epsilon) :
        Eold = Enew
        Fmat = F(a,c,alfa)
        eigvals, eigvecs = eigh(Fmat+Hmat, S, eigvals_only=False)
        Enew = np.array([eigvals[0],eigvals[1]])
        cnew = np.column_stack((eigvecs[:,0], eigvecs[:,1]))
        for i in range (0,2):
            for j in range (0,2):
                c[i,j] = cnew[i,j] * beta + (1-beta) * c[i,j]
        

cfinal1 = nor(c[:,0],alfa)
cfinal2 = nor(c[:,1],alfa)

        
#compute the total energy
uno = 0
due = 0
for p in range(0,2):
    for q in range(0,2):
            uno += (cfinal1[p]*cfinal1[q]+ cfinal2[p]*cfinal2[q])*Hmat[p,q]
            due += (cfinal1[p]*cfinal1[q]+ cfinal2[p]*cfinal2[q])*Fmat[p,q]
        
Etot1 = 2*uno+due
Etot2 = 2*(Enew[0]+Enew[1])-due 

# # now we plot
x = np.linspace(0,6,200)
cfinal1non = ([1.06668,1.28583,0.941678, 0.261637, -0.0123322, -0.00217616, 0.00339631, -0.000442856])
cfinal2non = ([0.213712, 0.272766, 0.27824, 0.160691, -0.0358804, -0.0320378, -0.0818085, -0.0790605])
alfanon = ([70.6486, 12.9278, 3.59149, 1.19198, 3.07283, 0.665203, 0.216283, 0.0830668])
f1snon = cfinal1non[0]* np.exp(-alfanon[0]*x**2) +cfinal1non[1]* np.exp(-alfanon[1]*x**2) +cfinal1non[2]* np.exp(-alfanon[2]*x**2) + cfinal1non[3]* np.exp(-alfanon[3]*x**2)+cfinal1non[4]* np.exp(-alfanon[4]*x**2)+cfinal1non[5]* np.exp(-alfanon[5]*x**2) + cfinal1non[6]* np.exp(-alfanon[6]*x**2)+cfinal1non[7]* np.exp(-alfanon[7]*x**2)
f2snon = cfinal2non[0]* np.exp(-alfanon[0]*x**2) +cfinal2non[1]* np.exp(-alfanon[1]*x**2) +cfinal2non[2]* np.exp(-alfanon[2]*x**2) + cfinal2non[3]* np.exp(-alfanon[3]*x**2)+cfinal2non[4]* np.exp(-alfanon[4]*x**2)+cfinal2non[5]* np.exp(-alfanon[5]*x**2) + cfinal2non[6]* np.exp(-alfanon[6]*x**2)+cfinal2non[7]* np.exp(-alfanon[7]*x**2)
# plot the function
f1s = -cfinal1[0]*(a[0]* np.exp(-alfa[0]*x**2) +a[1]* np.exp(-alfa[1]*x**2) +a[2]* np.exp(-alfa[2]*x**2) + a[3]* np.exp(-alfa[3]*x**2))- cfinal1[1]*(a[4]* np.exp(-alfa[4]*x**2)+a[5]* np.exp(-alfa[5]*x**2) + a[6]* np.exp(-alfa[6]*x**2)+a[7]* np.exp(-alfa[7]*x**2))
f2s = cfinal2[0]*(a[0]* np.exp(-alfa[0]*x**2) +a[1]* np.exp(-alfa[1]*x**2) +a[2]* np.exp(-alfa[2]*x**2) + a[3]* np.exp(-alfa[3]*x**2))+cfinal2[1]*(a[4]* np.exp(-alfa[4]*x**2)+a[5]* np.exp(-alfa[5]*x**2) + a[6]* np.exp(-alfa[6]*x**2)+a[7]* np.exp(-alfa[7]*x**2))
# plot the function
plt.figure()
plt.plot(x,f1snon,'r',color='green', label="1s orbital")
plt.plot(x,f2snon,'r',label= "2s orbital")
plt.plot(x,f1s,'r',color='blue', label="1s orbital contracted")
plt.plot(x,f2s,'r',color='black', label= "2s orbital contracted")
plt.legend()
plt.grid()
plt.ylabel('$\phi(r)$')
plt.xlabel('r')
plt.title("Beryllium 1s and 2s orbital")
plt.savefig("berillium.png", dpi=300)


density1s = 2*(abs(f1s)**2+abs(f1s)**2)*x**2*4*m.pi
density2s = 2*(abs(f2s)**2+abs(f2s)**2)*x**2*4*m.pi
density1snon = 2*(abs(f1snon)**2+abs(f1snon)**2)*x**2*4*m.pi
density2snon = 2*(abs(f2snon)**2+abs(f2snon)**2)*x**2*4*m.pi
plt.figure()
plt.plot(x,density1s,'r',color='blue',label= "1s contracted")
plt.plot(x,density2s,'r',color='black',label= "2s contracted")
plt.plot(x,density1snon,'r',color='green',label= "1s non contracted")
plt.plot(x,density2snon,'r',label= "2s non contracted")
plt.legend()
plt.grid()
plt.ylabel('$4\pi r^{2} \\rho(r)$')
plt.xlabel('r')
plt.title("Beryllium electronic radial distribution")
plt.savefig("Be_density.png", dpi=300)
 

