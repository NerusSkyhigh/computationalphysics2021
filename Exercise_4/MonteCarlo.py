#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:25:46 2022

@author: siriapasini
"""
import math as m
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from random import random
import AppKit

Np = 32 #number of particles
sigma = 0.2556 
n0 = 21.86 
epsilon = 10.22

xpos = np.zeros((Np))
ypos = np.zeros((Np))
zpos = np.zeros((Np))
dx = np.zeros([Np,Np])
dy = np.zeros([Np,Np])
dz = np.zeros([Np,Np])
r = np.zeros([Np,Np])
der = np.zeros([Np,Np])
der2 = np.zeros([Np,Np])


def distance(x,y,z):
    for i in range(0,Np):
        for j in range(0,Np):
            if i!=j:
                dx[i,j] = x[i]-x[j]
                dx[i,j] = dx[i,j] - L* np.rint(dx[i,j]/L)
                dy[i,j] = y[i]-y[j]
                dy[i,j] = dy[i,j] - L* np.rint(dy[i,j]/L)
                dz[i,j] = z[i]-z[j]
                dz[i,j] = dz[i,j] - L* np.rint(dz[i,j]/L)
                r[i,j] = (dx[i,j]**2 +dy[i,j]**2 +dz[i,j]**2)**0.5
                
    return r,dx,dy,dz


def probability(x, y,z, b):
    P = 0
    p = 0
    r,dx,dy,dz = distance(x,y,z)
    #print("{}".format(r))
    for i in range(0,Np):
        for j in range(0,i):
            if i!=j:
                p += (-b/r[i,j])**5

    P = np.exp(p)
    
    return P,r,dx,dy,dz

def sampling(xpos,ypos,zpos, delta, acc,rej, b):
    rx = np.zeros((Np))
    ry = np.zeros((Np))
    rz = np.zeros((Np))
    for i in range (0,Np):
        rx[i] = delta*(random()-0.5)
        ry[i] = delta*(random()-0.5)
        rz[i] = delta*(random()-0.5)
    xprop = xpos+ rx #generate proposed new positions
    xprop = xprop - L * np.rint(xprop/L)
    yprop = ypos+ry
    yprop = yprop - L * np.rint(yprop/L)
    zprop = zpos+rz
    zprop = zprop - L * np.rint(zprop/L)

    
    #evaluate ratio
    ratio = probability(xprop,yprop,zprop,b)[0]/ probability(xpos,ypos,zpos,b)[0]
    #print("{}".format(ratio))
    p = min(ratio,1)
    r2 = random()


    if p>r2:
        xpos = xprop
        ypos = yprop
        zpos = zprop
        acc += 1
    else: 
        rej += 1
    
    P,r,dx,dy,dz = probability(xpos,ypos,zpos,b)
    
    E, V,T, Tjf = energy(xpos, ypos, zpos, Np,b, r,dx,dy,dz)
    
    return xpos,ypos,zpos, acc, rej, E, V, T, Tjf   


def potential(r):
    V = 0
    for i in range(0,Np):
        for j in range(0,i):
            if i!=j:
                V += 4*((1/r[i,j])**12 - (1/r[i,j])**6)            
    return V

def kinetic(xpos,ypos,zpos,Np, b,r,dx,dy,dz):
    T =0
    Gx = np.zeros(Np)
    Gy = np.zeros(Np)
    Gz = np.zeros(Np)
    t = np.zeros(Np)    
    tjf = np.zeros(Np)
    Tjf=0

    for l in range (0,Np):
        for i in range (0,Np):
            if i!=l:                
                t[l] += 10*b**5 / (r[l,i]**7) 
                tjf[l] += 5*b**5 / (r[l,i]**7) 
        
     
            for k in range (0,Np):
                if k!=l and  i!=l:
                    Gx[l] += -(25/4) * b**10 * dx[l,i]*dx[l,k] / (r[l,i]**7 * r[l,k]**7)
                    Gy[l] += -(25/4) * b**10 * dy[l,i]*dy[l,k] / (r[l,i]**7 * r[l,k]**7)
                    Gz[l] += -(25/4) * b**10 * dz[l,i]*dz[l,k] / (r[l,i]**7 * r[l,k]**7)
        
        
        T += t[l] + Gx[l]+ Gy[l]+Gz[l] 
        Tjf += tjf[l]
        
    T = T * 0.09094
    Tjf = Tjf * 0.09094    
    
    return  T, Tjf


def energy(xpos,ypos,zpos,Np, b,r,dx,dy,dz):
    V = potential(r)
    T, Tjf = kinetic(xpos,ypos,zpos,Np,b,r,dx,dy,dz)
    E = V + T
    
    return E, V, T, Tjf

# INITIAL POSITION IN THE LATTICE (FOR NOW THIS IS OK, I USED THE ONE DEGIO DID, BUT I HAVE TO CHECK THIS AGAIN)
a = 0.5 * (Np / n0)**(1.0/3.0) /sigma
L = 2*a


n = 0

for k in range(0,4):
    for i in range(0,4):
        for j in range(0,2):
            zpos[n] = k * a / 2 # set z coordinate
            xpos[n] = i * a / 2 # set x coordinate
            if k%2 == 0:
        
                ypos[n] = j * a + i%2 * a/2 # set y coordinate
        
            else:
                
                ypos[n] = (j + 1/2) * a - i%2 * a/2 # set y coordinate
            

            n= n+1
            


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xpos,ypos,zpos, marker=".")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Initial Conditions")



# EQUILIBRATION STEP AND DETERMINATION OF DELTA
Ne = 1000 #equilibration number of steps
eff = 1
delta = 0.11 #initial choice just to have a starting point
b = 1.2

# Equilibration step
print("Starting equilibration and definition of delta")
while abs (eff - 0.5) > 0.1:
    acc = 0 #acceptance number
    rej = 0 #rejection number
    Esum = 0
    Esum2 = 0
    for j in range (0,Ne):
        xpos,ypos,zpos, acc, rej, E, V, T, Tjf  = sampling(xpos,ypos,zpos, delta,acc,rej,b)
        N = acc + rej
        eff = acc / N
        print("{}".format(j))
    delta = delta + 0.01

delta = delta - 0.01 
print("I found delta to be equal to {}".format(delta))  


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xpos,ypos,zpos, marker=".")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Actual Initial Conditions")


Nsteps = 10000

Nsamples = 1
I = np.zeros((Nsamples))
sigma = np.zeros((Nsamples))
b = np.zeros((Nsamples)) #variational parameter
b[0] = 1.2


for i in range (0,Nsamples):
    print("Value of b = {}".format(b[i]))
    acc = 0 #acceptance number
    rej = 0   
    Esum = 0
    Esum2 = 0
    E = np.zeros(Nsteps)
    V = np.zeros(Nsteps)
    T = np.zeros(Nsteps)
    Tjf = np.zeros(Nsteps)
    
    
    for j in range (0,Nsteps):
        xpos,ypos,zpos, acc, rej, E[j], V[j], T[j], Tjf[j]  = sampling(xpos,ypos,zpos, delta,acc,rej,b[i])
        N = acc + rej
        eff = acc / N
        Esum += E[j]
        Esum2 += E[j]**2
         
        print("{}".format(j))
        
    I[i] = Esum/Nsteps
    sigma[i] = (1/(Nsteps-1) * (Esum2/Nsteps - I**2) )**0.5
    print("{}".format(I[i]))

    
    if i != (Nsamples-1):
        b[i+1] = b[i]+ 0.1



fig = plt.figure()
plt.plot(np.linspace(0,Nsteps,Nsteps) , V, linewidth=0.9, label="V_{lj]")
plt.plot(np.linspace(0,Nsteps,Nsteps) , T, linewidth=0.9, label="T_{local}")
#plt.plot(np.linspace(0,Nsteps,Nsteps) , E, linewidth=0.9, label="E_{L}")
#plt.plot(np.linspace(0,Nsteps,Nsteps) , Tjf, linewidth=0.9, label="T_{JF,local}")
plt.legend()
plt.title("") 
plt.savefig("WF_20_K_el.png", dpi=300)
                
        
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(xpos,ypos,zpos, marker=".")
# # ax.set_xlabel("x")
# # ax.set_ylabel("y")
# # ax.set_zlabel("z")
# # plt.title("Final Conditions")   

# # AppKit.NSBeep()

