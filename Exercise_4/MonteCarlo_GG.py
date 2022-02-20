#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math as m
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from random import random
#import AppKit

Np = 32 #number of particles
sigma = 0.2556
n0 = 21.86
epsilon = 10.22

# Position of each particle
xpos = np.zeros(Np); ypos = np.zeros(Np); zpos = np.zeros(Np)

# Distance of each particle with each other particle
dx = np.zeros([Np,Np]); dy = np.zeros([Np,Np]); dz = np.zeros([Np,Np])

# Idk
r = np.zeros([Np,Np])
der = np.zeros([Np,Np])
der2 = np.zeros([Np,Np])


def distance(x,y,z):
    for i in range(0,Np):
        # As the distances are symmetric
        # I can iterate over half particles
        # dx[i,i] = dy[i,i] = dz[i,i] = 0
        for j in range(0,i):
            dx[i,j] = x[i]-x[j]
            dx[i,j] = dx[i,j] - L*np.rint(dx[i,j]/L) # pbc

            dy[i,j] = y[i]-y[j]
            dy[i,j] = dy[i,j] - L*np.rint(dy[i,j]/L)

            dz[i,j] = z[i]-z[j]
            dz[i,j] = dz[i,j] - L*np.rint(dz[i,j]/L)

            # Exploit symmetric form
            dx[j,i] = dx[i,j]

    # Numpy already computes this opertion element by element.
    # If the processor supports matrix multiplication this is faster
    r = np.sqrt(np.power(dx, 2) +
                np.power(dy, 2) +
                np.power(dz, 2) )

    return r, dx, dy, dz


def probability(x, y, z, b):
    p = 0
    r, dx, dy, dz = distance(x,y,z)

    #print("{}".format(r))
    for i in range(0,Np):
        for j in range(0,i):
            p += (-b/r[i,j])**5

    return np.exp(p), r, dx, dy, dz


def sampling(xpos, ypos, zpos, delta, acc, rej, b):
    # Generate variations
    rx = delta*(np.random.random_sample(Np)-0.5)
    ry = delta*(np.random.random_sample(Np)-0.5)
    rz = delta*(np.random.random_sample(Np)-0.5)

    # Generate proposed new positions (and impose pbc)
    xprop = xpos+rx; xprop = xprop - L*np.rint(xprop/L)
    yprop = ypos+ry; yprop = yprop - L*np.rint(yprop/L)
    zprop = zpos+rz; zprop = zprop - L*np.rint(zprop/L)


    # Probability is a deterministic function so i can
    # avoitd computing it 3 times and compute it only once
    # prop = for the variables "proposte"
    P,      r,      dx,      dy,      dz      = probability(xpos,  ypos,  zpos, b)
    P_prop, r_prop, dx_prop, dy_prop, dz_prop = probability(xprop, yprop, zprop,b)

    # Evaluate ratio

    ratio = P_prop / P
    p = min(ratio, 1)

    # Qui è corretto che sia così o serve un loop?
    r2 = random()
    if p>r2:
        # I accept the proposal. Your ring was fine
        acc += 1
        
        # Update positions, probability and distances
        xpos = xprop; ypos = yprop; zpos = zprop
        P = P_prop;
        r = r_prop; dx = dx_prop; dy = dy_prop; dz = dz_prop
    else:
        # Proposal denied. A sapphire ring is not enough
        rej += 1

    # I don't need to compute this once again.
    # I saved them from before
    #P,r,dx,dy,dz = probability(xpos,ypos,zpos,b)

    E, V,T, Tjf = energy(xpos, ypos, zpos, Np,b, r, dx,dy,dz)

    # Maybe we can save the values even between iterations?
    # But I'm worried that numba might protest
    #old_probability = { 'P': P,
    #                    'r': r,
    #                    'dx':dx, 'dy':dy, 'dz':dz}

    return xpos,ypos,zpos, acc, rej, E, V, T, Tjf


def potential(r):
    V = 0
    for i in range(0,Np):
        for j in range(0,i):
            if i!=j:
                V += 4*((1/r[i,j])**12 - (1/r[i,j])**6)
    return V

def kinetic(xpos,ypos,zpos,Np, b,r,dx,dy,dz):
    Gx = np.zeros(Np)
    Gy = np.zeros(Np)
    Gz = np.zeros(Np)
    t = np.zeros(Np)
    tjf = np.zeros(Np)

    #for l in range(0,Np):
    #    for i in range(0,Np):
    #        if i!=l:
    #            t[l]+=  10*b**5 / (r[l,i]**7)
    #            tjf[l]+= 5*b**5 / (r[l,i]**7)
    for l in range(0,Np): # I retyped the double loop
        for i in range(0,l):
            # The first factor 2 is to include the
            # double summation r[l,i] = r[i, l]
            tjf[l]+= 2*(5*b**5 / (r[l,i]**7))
    t = tjf*2 #It's the same formula

    for l in range (0,Np):
        for i in range(0,l):
            for k in range (0,Np):
                if k!=l and  i!=l:
                    # I don't think this can be optimized due to dx, dy, dy
                    # there are no symetries to exploit
                    Gx[l] += dx[l,i]/(r[l,i]**7) * dx[l,k]/(r[l,k]**7)
                    Gy[l] += dy[l,i]*dy[l,k] / (r[l,i]**7 * r[l,k]**7)
                    Gz[l] += dz[l,i]*dz[l,k] / (r[l,i]**7 * r[l,k]**7)
    Gx = -(25/4)*b**10 * Gx
    Gy = -(25/4)*b**10 * Gy
    Gz = -(25/4)*b**10 * Gz

    # np.sum automatically sums an array
    T = 0.09094* np.sum( t + Gx + Gy +Gz)
    Tjf = 0.09094* np.sum(tjf)

    return  T, Tjf


def energy(xpos,ypos,zpos,Np, b,r,dx,dy,dz):
    V = potential(r)
    T, Tjf = kinetic(xpos,ypos,zpos,Np,b,r,dx,dy,dz)
    E = V + T

    return E, V, T, Tjf



################
#  MAIN LOOP   #
################
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

            n=n+1



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
while abs(eff - 0.5) > 0.1:
    acc = 0 #acceptance number
    rej = 0 #rejection number
    Esum = 0
    Esum2 = 0
    for j in range (0,Ne):
        xpos,ypos,zpos, acc, rej, E, V, T, Tjf  = sampling(xpos,ypos,zpos, delta,acc,rej,b)
        N = acc + rej
        eff = acc / N
        
        if(j % 10 == 0):
            print("Equilibrium {}%".format(j/Ne*100))
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

        if(j%100 == 0):
            print("Steps {}%".format(j/Nsteps*100))

    I[i] = Esum/Nsteps
    sigma[i] = (1/(Nsteps-1) * (Esum2/Nsteps - I**2) )**0.5
    print("{}".format(I[i]))


    if i != (Nsamples-1):
        b[i+1] = b[i]+ 0.1




################
#  MAIN LOOP   #
################

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
