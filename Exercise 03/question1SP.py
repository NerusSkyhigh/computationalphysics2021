# Densities

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt

# Define system parameters

Ne = 18 # Number of electrons. 

#rs = 3.93 # For Na 
rs = 4.86 # For K

rhob = 3.0 / (4.0 * m. pi * rs**3) 

Rc = (3.0 * Ne / (4.0 * m. pi * rhob) )**(1.0 / 3.0)

# Setting some initial parameters 

Nx = 1000 #number of mesh points
dE = 0.0001 # energy step
L = 17.0 # Mesh length
dx = L / Nx # mesh spacing
x = np.arange(0, dx * Nx, dx)

# Define a function for the potential 

@njit
def potential(x, l):
    
    if x <= Rc:
        
        V = 2.0 * m.pi * rhob * ( (1.0 / 3.0) * x**2 - Rc**2 )
        
    else:
        
        V = - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / x)
        
    if x != 0:
        
        V += l*(l+1) / (2 * x**2)

    return V

# Define a coulomb function for the drawing phase

@njit
def coulomb(x):
    
    if x <= Rc:
        
        V = 2.0 * m.pi * rhob * ( (1.0 / 3.0) * x**2 - Rc**2 )
        
    else:
        
        V = - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / x)

    return V

# Define Numerov function 

@njit
def numerov(psi1, psi2, i, E, l):
    
    p1 = potential(dx * (i-2), l)
    p2 = potential(dx * (i-1), l)
    pf = potential(dx * i, l)
    
    k1 = 2.0 * (E - p1)
    k2 = 2.0 * (E - p2)
    kf = 2.0 * (E - pf)

    num1 =  psi2 * (2.0 - (5.0 / 6.0) * k2 * dx**2) 
    num2 = psi1 * (1.0 + (1.0 / 12.0) * k1 * dx**2)
    den = 1.0 + (1.0 / 12.0) * kf * dx**2

    psif = (num1 - num2) / den
    
    return psif
     

# Function to construct the density

@njit 
def rhoconstr(psi): #psi is the radial function in this case
    
    rho = np.zeros(Nx)
    
    for i in range(1,Nx,1):
        
        rho[i] = (psi[i] / x[i])**2
        
    rho[0] = rho[1]

    
    return rho 

def normalisation_density(rho,N):
    I = 0
    for i in range(1,Nx,1):
        I += rho[i]*dx#*4*m.pi*x[i]**2
        
    rho= rho/I *N
    
    return rho
# Function to integrate for normalisation

@njit
def normalisation(x, psi): #normalized to number of electrons
    
    I = 0
    for i in range(0,int(14/dx)):

        I += psi[i]*psi[i]*dx
    
    psi = psi /I**0.5
     
    return psi

        
psi = np.zeros(Nx)
Ener = - 2.0 * m.pi * rhob * Rc**2+dE #set the starting energy at the bottom of the well
Es = np.zeros((4,2))

# for l in range(0,1):
#     for n in range(l+1,l+4):
for l in range(0,2):
    lastnow = 1
    if l==1:
        Ener = Es[1,0] +dE #the energy level of the 2p must be higher than the 2s
        lastnow = -1 #this is set like this according to the last value of the function, since it has 1 node we expect its last value to be negative
    
    for n in range(l,4):
        if l == 0:
            Nel = 2
        elif l == 1:
            Nel = 6
        
        psi[0] = 0
        psi[1] = pow(dx, l+1)
            
        
        lastbefore = 1
        rat = 1.0
    
        while rat > 0:
                   
            for i in range(2, Nx, 1):
                   
                psi[i] = numerov(psi[i-2], psi[i-1], i, Ener, l)
                   
           
            lastbefore = lastnow 
            lastnow = psi[Nx-1]
            rat = lastnow / lastbefore
               
            Ener = Ener + dE    
            
        Es[n,l] = Ener - dE - (dE / (lastnow - lastbefore) ) * lastnow
        # Ener = Es[n,l]+dE
        lastnow = psi[Nx-1]

# # to check
# psitest1s = np.zeros(Nx)
# psitest1s[0] = 0        
# psitest1s[1] = pow(dx, l+1)

# for i in range(2, Nx, 1):
#      psitest1s[i] = numerov(psitest1s[i-2], psitest1s[i-1], i, Es[1,0] , 0)
     
# psitest1spre = np.zeros(Nx)
# psitest1spre[0] = 0        
# psitest1spre[1] = pow(dx, l+1)
# for i in range(2, Nx, 1):
#      psitest1spre[i] = numerov(psitest1spre[i-2], psitest1spre[i-1], i, (Es[1,0]-dE), 0)

psi1s = np.zeros(Nx)
psi1s[0] = 0        
psi1s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
      psi1s[i] = numerov(psi1s[i-2], psi1s[i-1], i, Es[0,0] , 0)
      
psi2s = np.zeros(Nx)
psi2s[0] = 0        
psi2s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
      psi2s[i] = numerov(psi2s[i-2], psi2s[i-1], i, Es[1,0] , 0)
      
psi3s = np.zeros(Nx)
psi3s[0] = 0        
psi3s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
      psi3s[i] = numerov(psi3s[i-2], psi3s[i-1], i, Es[2,0] , 0)
      
psi4s = np.zeros(Nx)
psi4s[0] = 0        
psi4s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
      psi4s[i] = numerov(psi4s[i-2], psi4s[i-1], i, Es[3,0] , 0)
      
psi2p = np.zeros(Nx)
psi2p[0] = 0        
psi2p[1] = pow(dx, 2)
for i in range(2, Nx, 1):
      psi2p[i] = numerov(psi2p[i-2], psi2p[i-1], i, Es[1,1] , 1)
      
psi3p = np.zeros(Nx)
psi3p[0] = 0        
psi3p[1] = pow(dx, 2)
for i in range(2, Nx, 1):
      psi3p[i] = numerov(psi3p[i-2], psi3p[i-1], i, Es[2,1] , 1)



#with the correct normalization
psi1s = normalisation(x, psi1s)
psi2s= normalisation(x, psi2s)
psi3s= normalisation(x, psi3s)
psi4s= normalisation(x, psi4s)
psi2p= normalisation(x, psi2p)
psi3p= normalisation(x, psi3p)


#building the density
rho1s = normalisation_density(rhoconstr(psi1s),2)
rho2s = normalisation_density(rhoconstr(psi2s),2)
rho3s = normalisation_density(rhoconstr(psi3s),2)
rho4s = normalisation_density(rhoconstr(psi4s),2)
rho2p = normalisation_density(rhoconstr(psi2p),6)
rho3p = normalisation_density(rhoconstr(psi3p),6)

#denisty plot
rho = rho1s + rho2s + rho2p + rho3s + rho3p
#rho = normalisation_density(rho,10)

I=0
for i in range(1,Nx,1):
    I += rho[i] * dx#*4*m.pi*x[i]**2

fig1 = plt.figure()
plt.plot(x[0:int(17/dx)], rho[0:int(17/dx)], linewidth=0.9, label = "total")
# plt.plot(x[0:int(14/dx)], rho1s[0:int(14/dx)], linewidth=0.9, label = "1s")
# plt.plot(x[0:int(14/dx)], rho2s[0:int(14/dx)], linewidth=0.9, label = "2s")
# plt.plot(x[0:int(14/dx)], rho3s[0:int(14/dx)], linewidth=0.9, label = "3s")
# # plt.plot(x[0:int(14/dx)], rho4s[0:int(14/dx)], linewidth=0.9, label = "4s")
# plt.plot(x[0:int(14/dx)], rho2p[0:int(14/dx)], linewidth=0.9, label = "2p")
# plt.plot(x[0:int(14/dx)], rho3p[0:int(14/dx)], linewidth=0.9, label = "3p")
plt.legend()
plt.title("Electron density for K")
# plt.axvline(Rc, color = "grey")
plt.savefig("rho_19_el.png", dpi=300)

# WF figure
fig = plt.figure()
plt.plot(x[0:int(17/dx)], psi1s[0:int(17/dx)], linewidth=0.9, label="1s")
plt.plot(x[0:int(17/dx)], psi2s[0:int(17/dx)], linewidth=0.9, label ="2s")
plt.plot(x[0:int(17/dx)], psi2p[0:int(17/dx)], linewidth=0.9, label="2p")
plt.plot(x[0:int(17/dx)], psi3p[0:int(17/dx)], linewidth=0.9, label="3p")
plt.plot(x[0:int(17/dx)], psi3s[0:int(17/dx)], linewidth=0.9, label="3s")
# plt.plot(x[0:int(14/dx)], psi4s[0:int(14/dx)], linewidth=0.9, label="4s")
plt.legend()
plt.title("Wavefunctions")
#plt.axvline(Rc, color = "grey")
plt.savefig("WF_19_el.png", dpi=300)

#figure of potential with levels
V0 = np.zeros(Nx)

for i in range(0, Nx):
    V0[i] = coulomb(x[i])  

fig2 = plt.figure()
plt.plot(x, V0, linewidth=0.9)
plt.title("Potential")
plt.axhline(Es[0,0] , color="green", linestyle="-.", label="1s", linewidth=0.9)
plt.axhline(Es[1,0] , color="blue", linestyle="-.", label="2s", linewidth=0.9)
plt.axhline(Es[1,1] , color="orange", linestyle="-.", label="2p", linewidth=0.9)
plt.axhline(Es[2,0] , color="red", linestyle="-.", label="3s", linewidth=0.9)
plt.axhline(Es[2,1] , color="cyan", linestyle="-.", label="3p", linewidth=0.9)
plt.axhline(Es[3,0] , color="black", linestyle="-.", label="4s", linewidth=0.9)
plt.axhline(Es[3,1] , color="brown", linestyle="-.", label="4p", linewidth=0.9)
plt.legend()
plt.savefig("Pot_20_el.png", dpi=300)   
    
    
    