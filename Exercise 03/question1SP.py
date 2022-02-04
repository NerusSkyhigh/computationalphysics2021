# Densities

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Define system parameters

Ne = 20 # Number of electrons. 

#rs = 3.93 # For Na 
rs = 4.86 # For K

rhob = 3.0 / (4.0 * m. pi * rs**3) 

Rc = (Ne)**(1.0 / 3.0) * rs

# Setting some initial parameters 

Nx = 1000 #number of mesh points
dE = 0.0001 # energy step
L = 15 # Mesh length
dx = L / Nx # mesh spacing
x = np.arange(0, dx * Nx, dx)

Ldraw = 14
Nxx = int(Ldraw/dx)
# Define a function for the potential 

@njit
def potential(x, l):
     #average coulomb term between electrons
    
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
def rhoconstruct(R,x,l):
    
    rho = np.zeros(Nx)

    for i in range (1,Nx,1): 
        rho[i] = R[i] *(2*l+1)/(2*m.pi) 
    
    return rho
 
        
def level(n,l,Ener,lastnow):
    u = np.zeros(Nx)
    #Ener = - 2.0 * m.pi * rhob * Rc**2+dE #set the starting energy at the bottom of the well
    #lastnow = 1
    u[0] = 0
    u[1] = pow(dx, l+1)
    lastbefore = 1
    rat = 1.0
    
    while rat > 0:
                   
        for i in range(2, Nx, 1):
                   
            u[i] = numerov(u[i-2], u[i-1], i, Ener, l)
                   
           
        lastbefore = lastnow 
        lastnow = u[Nx-1]
        rat = lastnow / lastbefore
           
        Ener = Ener + dE    
        
    Ener = Ener - dE - (dE / (lastnow - lastbefore) ) * lastnow
    # Ener = Es[n,l]+dE
    lastnow = u[Nx-1]
    
    return u, Ener, lastnow

#building the levels
E = np.zeros((4,4))
u1s, E[0,0],lastnow1s = level(1,0, - 2.0 * m.pi * rhob * Rc**2+dE, 1)
u1p, E[0,1],lastnow1p = level(1,1, E[0,0]+dE , 1)
u1d, E[0,2],lastnow1d = level(1,2, E[0,1]+dE , 1)
u2s, E[1,0],lastnow2s = level(2,0, E[0,2]+5*dE , -1)
R1s = np.zeros(Nx)
R1p = np.zeros(Nx)
R1d = np.zeros(Nx)
R2s = np.zeros(Nx)



u1s[0] = 0
u1s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
    u1s[i] =  numerov(u1s[i-2], u1s[i-1], i, E[0,0], 0)
    
Iu = np.trapz(u1s**2, x )
u1s = u1s / Iu**0.5

u1p[0] = 0
u1p[1] = pow(dx, 2)
for i in range(2, Nx, 1):
    u1p[i] = numerov(u1p[i-2], u1p[i-1], i, E[0,1], 1)  
Iu = np.trapz(u1p**2, x )
u1p = u1p / Iu**0.5


u1d[0] = 0
u1d[1] = pow(dx, 3)
for i in range(2, Nx, 1):
    u1d[i] = numerov(u1d[i-2], u1d[i-1], i, E[0,2], 2)
Iu = np.trapz(u1d**2, x )
u1d = u1d / Iu**0.5

u2s[0] = 0
u2s[1] = pow(dx, 1)
for i in range(2, Nx, 1):
    u2s[i] = numerov(u2s[i-2], u2s[i-1], i, E[1,0], 0)
Iu = np.trapz(u2s**2, x )
u2s = u2s / Iu**0.5   

for i in range(1,Nx):        
    R1s[i] = (u1s[i]/x[i])**2
    R1p[i] = (u1p[i]/x[i])**2 
    R1d[i] = (u1d[i]/x[i])**2
    R2s[i] = (u2s[i]/x[i])**2




# WF figure
fig = plt.figure()
plt.plot(x[1:int(L/dx)], R1s[1:int(L/dx)], linewidth=0.9, label="1s")
plt.plot(x[1:int(L/dx)], R1p[1:int(L/dx)], linewidth=0.9, label ="1p")
plt.plot(x[1:int(L/dx)], R1d[1:int(L/dx)], linewidth=0.9, label="1d")
plt.plot(x[1:int(L/dx)], R2s[1:int(L/dx)], linewidth=0.9, label="2s")
# plt.plot(x[0:int(L/dx)], psi3s[0:int(L/dx)], linewidth=0.9, label="3s")
# plt.plot(x[0:int(14/dx)], psi4s[0:int(14/dx)], linewidth=0.9, label="4s")
plt.legend()
plt.title("Wavefunctions")
#plt.axvline(Rc, color = "grey")
plt.savefig("WF_19_el.png", dpi=300)



rho = rhoconstruct(R1s, x, 0) + rhoconstruct(R1p, x, 1) + rhoconstruct(R1d, x, 2) +  rhoconstruct(R2s, x, 0)

Ir = np.trapz(rho*x**2*4*m.pi, x )


V0 = np.zeros(Nx)

for i in range(1, Nx):
    V0[i] = coulomb(x[i])



fig1 = plt.figure()
plt.plot(x[1:int(L/dx)], rho[1:int(L/dx)], linewidth=0.9, label = "total")
# plt.plot(x[2:int(Ldraw/dx)], normalisation(x, rhoconstruct(u1s, x, 0),0)[2:int(Ldraw/dx)], linewidth=0.9, label = "1s")
# plt.plot(x[2:int(Ldraw/dx)], normalisation(x, rhoconstruct(u1p, x, 1),1)[2:int(Ldraw/dx)], linewidth=0.9, label = "1p")
# plt.plot(x[2:int(Ldraw/dx)], normalisation(x, rhoconstruct(u1d, x, 2),2)[2:int(Ldraw/dx)], linewidth=0.9, label = "1d")
# plt.plot(x[2:int(Ldraw/dx)], normalisation(x, rhoconstruct(u2s, x, 0),0)[2:int(Ldraw/dx)], linewidth=0.9, label = "2s")
# plt.plot(x[0:int(14/dx)], rho2p[0:int(14/dx)], linewidth=0.9, label = "2p")
#plt.plot(x[2:int(L/dx)], V0[2:int(L/dx)], linewidth=0.9)
# # plt.plot(x[0:int(14/dx)], rho3p[0:int(14/dx)], linewidth=0.9, label = "3p")
plt.legend()
plt.title("Electron density for Na")
plt.axvline(Rc, color = "grey")
plt.savefig("rho_19_el.png", dpi=300)


# # WF figure
# fig = plt.figure()
# plt.plot(x[2:int(L/dx)], u1s[2:int(L/dx)], linewidth=0.9, label="1s")
# plt.plot(x[2:int(L/dx)], u1p[2:int(L/dx)], linewidth=0.9, label ="1p")
# plt.plot(x[2:int(L/dx)], u1d[2:int(L/dx)], linewidth=0.9, label="1d")
# plt.plot(x[2:int(L/dx)], u2s[2:int(L/dx)], linewidth=0.9, label="2s")
# # plt.plot(x[0:int(L/dx)], psi3s[0:int(L/dx)], linewidth=0.9, label="3s")
# # plt.plot(x[0:int(14/dx)], psi4s[0:int(14/dx)], linewidth=0.9, label="4s")
# plt.legend()
# plt.title("Wavefunctions")
# #plt.axvline(Rc, color = "grey")
# plt.savefig("WF_19_el.png", dpi=300)

# #figure of potential with levels
# V0 = np.zeros(Nx)

# for i in range(1, Nx):
#     V0[i] = coulomb(x[i])  

# fig2 = plt.figure()
# plt.plot(x[2:int(L/dx)], V0[2:int(L/dx)], linewidth=0.9)
# plt.title("Potential")
# plt.axhline(E[0,0] , color="green", linestyle="-.", label="1s", linewidth=0.9)
# plt.axhline(E[0,1] , color="blue", linestyle="-.", label="1p", linewidth=0.9)
# plt.axhline(E[0,2] , color="orange", linestyle="-.", label="1d", linewidth=0.9)
# plt.axhline(E[1,0] , color="red", linestyle="-.", label="1f", linewidth=0.9)
# # plt.axhline(E[1,0] , color="cyan", linestyle="-.", label="2s", linewidth=0.9)
# # plt.axhline(E[1,1] , color="black", linestyle="-.", label="2p", linewidth=0.9)
# # plt.axhline(E[1,2] , color="brown", linestyle="-.", label="2d", linewidth=0.9)
# plt.legend()
# plt.savefig("Pot_20_el.png", dpi=300)   
    
    
    