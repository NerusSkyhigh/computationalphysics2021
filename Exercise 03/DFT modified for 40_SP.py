# DFT 

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import AppKit




# Define system parameters

Ne = 40 # Number of electrons. Try with 2, 8, 18, 20

rs = 3.93 # For Na 
#rs = 4.86 # For K

rhob = 3.0 / (4.0 * m. pi * rs**3) 

Rc = (3.0 * Ne / (4.0 * m. pi * rhob) )**(1.0 / 3.0)

# Setting some initial parameters 

Nx = 1000 #number of mesh points
#dE = 0.0001# energy step
#dE = 0.1
L = 18.5 # Mesh length  (for 8 electrons until 15 is fine, for 20 electrons til 17.5 is ok with 0.00005 I think )
dx = L / Nx # mesh spacing
x = np.arange(0, dx * Nx, dx) 


# Correlation function parameters

p = 1.0
A = 0.031091
a1 = 0.21370
b1 = 7.5957
b2 = 3.5876
b3 = 1.6382
b4 = 0.49294

# Mixing parameter

a = 0.2

# Define a function that computes the direct term in the potential

@njit 
def direct(r, rho):
    rho1 = np.zeros(Nx)
    rho2 = np.zeros(Nx)
    for i in range(0,int(r/dx)):
        rho1[i] = rho[i]
    for j in range(int(r/dx),Nx):
        rho2[j] = rho[j]
    
    if r!=0:
        d = 4*m.pi* (np.trapz(rho1 * x**2, x)/r + np.trapz(rho2 * x, x))
    else:
        d = 4*m.pi* np.trapz(rho2 * x, x)
    
    return d


# Define a function for the potential. Takes as input, position r, l quantum # and mixed density

@njit
def potential(r, l, rho, rhopoint):

    V = direct(r, rho)  -2.0 * A * (1 + a1 * rs) * m.log(1.0 + 1.0 / (2.0 * A * (b1 * rs**(1.0/2.0) + b2 * rs + b3 * rs**(3.0/2.0) + b4 * rs**(p + 1.0) ) ) ) - (3*rhopoint/m.pi )**(1/3)
    
    if r <= Rc:
        
        V += 2.0 * m.pi * rhob * ( (1.0 / 3.0) * r**2 - Rc**2 )
        
    else:
        
        V += - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / r)
        
    if r != 0:
        
        V += l * (l + 1) / (2 * r**2)

    return V

# Define a potential function for the drawing phase
# Takes as input position r and mixed density rho

@njit
def draw_pot(r, rho, rhopoint):
    
    V = direct(r, rho) - 2.0 * A * (1 + a1 * rs) * m.log(1.0 + 1.0 / (2.0 * A * (b1 * rs**(1.0/2.0) + b2 * rs + b3 * rs**(3.0/2.0) + b4 * rs**(p + 1.0) ) ) ) - (3/m.pi *rhopoint)**(1/3)
    
    if r <= Rc:
        
        V += 2.0 * m.pi * rhob * ( (1.0 / 3.0) * r**2 - Rc**2 )
        
    else:
        
        V += - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / r)

    return V

# Define Numerov function 

@njit
def numerov(psi1, psi2, i, E, l, rho, rhof,rho2,rho1):
    
    p1 = potential(dx * (i-2), l, rho, rho1)
    p2 = potential(dx * (i-1), l, rho, rho2)
    pf = potential(dx * i, l, rho, rhof)
    
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
        rho[i] = R[i] *2*(2*l+1)/(4*m.pi) 
    
    return rho

@njit 
def drawvec(rhomix):
    
    V0 = np.zeros(Nx)
    di = np.zeros(Nx)
    co = np.zeros(Nx)
    ex = np.zeros(Nx)

    for i in range(0, Nx):
        V0[i] = draw_pot(x[i], rhomix, rhomix[i])
        di[i] = direct(x[i], rhomix)
        co[i] = -2.0 * A * (1 + a1 * rs) * m.log(1.0 + 1.0 / (2.0 * A * (b1 * rs**(1.0/2.0) + b2 * rs + b3 * rs**(3.0/2.0) + b4 * rs**(p + 1.0) ) ) )
        ex[i] = -(3/m.pi *rhomix[i])**(1/3)
        
    return V0, di, co, ex



def level(n,l,Ener,lastnow, rho, dE):
    u = np.zeros(Nx)
    #Ener = - 2.0 * m.pi * rhob * Rc**2+dE #set the starting energy at the bottom of the well
    #lastnow = 1
    u[0] = 0
    u[1] = pow(dx, l+1)
    lastbefore = 1
    rat = 1.0
    
    while rat > 0:
                   
        for i in range(2, Nx, 1):
                   
            u[i] = numerov(u[i-2], u[i-1], i, Ener, l, rho, rho[i], rho[i-1], rho[i-2])
                   
           
        lastbefore = lastnow 
        lastnow = u[Nx-1]
        rat = lastnow / lastbefore
           
        Ener = Ener + dE   
        #print("{}".format(Ener-dE))
        #print("{}".format(rat))
    
    Ener = Ener -  2*dE #- (dE / (lastnow - lastbefore) ) * lastnow
    for i in range(2, Nx, 1):
        u[i] =  numerov(u[i-2], u[i-1], i, Ener, 0, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
    Iu = np.trapz(u**2, x )
    u = u / Iu**0.5   
    lastnow = u[Nx-1]
    
    #print("{}".format(Ener))
    #print("{}".format(lastnow))
    
    if lastnow >0.0001:
        
        print("refinement")
        j = 0
        
        while (lastnow >0.0001 or lastnow < -0.0001):
            rat = 1
            lastbefore = lastnow
            u[0] = 0
            u[1] = pow(dx, l+1)
            dEref = 0.001
            if j!=0:
                dEref = 0.001/(10**j)
            #lastnow = 1
            while rat > 0:
                           
                for i in range(2, Nx, 1):
                           
                    u[i] = numerov(u[i-2], u[i-1], i, Ener, l, rho, rho[i], rho[i-1], rho[i-2])
                    
                lastbefore = lastnow 
                lastnow = u[Nx-1]
                rat = lastnow / lastbefore
                   
                Ener = Ener + dEref
                #print("{}".format(Ener-dE))
                #print("{}".format(lastnow))
            
            Ener = Ener - 2*dEref
            # Ener = Es[n,l]+dE
            lastnow = u[Nx-1]
            j +=1
            print("{}".format(j))
            
        Ener = Ener + dEref #- (dE / (lastnow - lastbefore) ) * lastnow 
        print("final value of the energy{}".format(Ener))
        # Ener = Es[n,l]+dE
        lastnow = u[Nx-1]
        
    else:
        print("no refinement")
        #Ener = Ener + dE
        print("final value of the energy{}".format(Ener))
        # Ener = Es[n,l]+dE
        #lastnow = u[Nx-1]
        
        
    return u, Ener, lastnow


# SELF CONSISTENT PROCEDURE 


epsilon = 0.02
    
# Set initial densities
    
rhold = np.zeros(Nx)
rhoinitial =  np.zeros(Nx)
rhofinal = np.zeros(Nx)
    
Eold = np.zeros((2,4))
Enew = np.ones((2,4))

j= 0
 #set the starting energy at the bottom of the well

while (abs(Enew[0,0]-Eold[0,0]) > epsilon) or (abs(Enew[0,1]-Eold[0,1]) > epsilon) or (abs(Enew[0,2]-Eold[0,2]) > epsilon) or (abs(Enew[1,0]-Eold[1,0]) > epsilon or (abs(Enew[0,3]-Eold[0,3])) > epsilon or (abs(Enew[1,1]-Eold[1,1])) > epsilon):
#while  (all(abs(rhold - rhonew))) > epsilon or (all(abs(rhold - rhonew))) == 0: # or (abs(Enew[1,0]-Eold[1,0]) > epsilon) :
#for j in range(0,18):
    print("{}".format(j))
    rhomix = rhoinitial
    V0, di, ex, co = drawvec(rhomix)
    minE = min(V0)
    #dE = dE/((j+1)*100)
    dE = 0.01
    #if j>2:
     #   dE = 0.0001/(j*10)

    
    u = np.zeros(Nx)
    E = np.zeros((2,4))
            
    u1s, E[0,0],lastnow1s = level(1,0, minE , 1, rhomix, dE)
    print("First level done")
    u1p, E[0,1],lastnow1p = level(1,1, E[0,0] , 1, rhomix,dE)
    print("Second level done")
    u1d, E[0,2],lastnow1d = level(1,2, E[0,1], 1, rhomix,dE)
    print("Third level done")
    u2s, E[1,0],lastnow2s = level(2,0, E[0,2] , -1, rhomix,dE)
    print("Fourth level done")
    u1f, E[0,3],lastnow1f = level(1,3, E[1,0], 1, rhomix,dE)
    print("Fifth level done")
    u2p, E[1,1],lastnow2p = level(2,1, E[0,3], 1, rhomix,dE)
    print("Sixth level done")
    print("{},{},{},{},{},{}".format(E[0,0],E[0,1],E[0,2],E[1,0],E[0,3],E[1,1]))
    R1s = np.zeros(Nx)
    R1p = np.zeros(Nx)
    R1d = np.zeros(Nx)
    R2s = np.zeros(Nx)
    R1f = np.zeros(Nx)
    R2p = np.zeros(Nx)
    

    u1s[0] = 0
    u1s[1] = pow(dx, 1)
    for i in range(2, Nx, 1):
        u1s[i] =  numerov(u1s[i-2], u1s[i-1], i, E[0,0], 0, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
        
    Iu = np.trapz(u1s**2, x )
    u1s = u1s / Iu**0.5
    
    u1p[0] = 0
    u1p[1] = pow(dx, 2)
    for i in range(2, Nx, 1):
        u1p[i] = numerov(u1p[i-2], u1p[i-1], i, E[0,1], 1, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])  
    Iu = np.trapz(u1p**2, x )
    u1p = u1p / Iu**0.5
    
    
    u1d[0] = 0
    u1d[1] = pow(dx, 3)
    for i in range(2, Nx, 1):
        u1d[i] = numerov(u1d[i-2], u1d[i-1], i, E[0,2], 2, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
    Iu = np.trapz(u1d**2, x )
    u1d = u1d / Iu**0.5
    
    u2s[0] = 0
    u2s[1] = pow(dx, 1)
    for i in range(2, Nx, 1):
        u2s[i] = numerov(u2s[i-2], u2s[i-1], i, E[1,0], 0, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
    Iu = np.trapz(u2s**2, x )
    u2s = u2s / Iu**0.5   
    
    u2p[0] = 0
    u2p[1] = pow(dx, 2)
    for i in range(2, Nx, 1):
        u2p[i] = numerov(u2p[i-2], u2p[i-1], i, E[1,1], 1, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
    Iu = np.trapz(u2p**2, x )
    u2p = u2p / Iu**0.5   
    
    u1f[0] = 0
    u1f[1] = pow(dx, 4)
    for i in range(2, Nx, 1):
        u1f[i] =  numerov(u1f[i-2], u1f[i-1], i, E[0,3], 3, rhomix, rhomix[i], rhomix[i-1],rhomix[i-2])
        
    Iu = np.trapz(u1f**2, x )
    u1f = u1f / Iu**0.5
    
    
    for i in range(1,Nx):        
        R1s[i] = (u1s[i]/x[i])**2
        R1p[i] = (u1p[i]/x[i])**2 
        R1d[i] = (u1d[i]/x[i])**2
        R2s[i] = (u2s[i]/x[i])**2
        R2p[i] = (u2p[i]/x[i])**2
        R1f[i] = (u1f[i]/x[i])**2


       
    rhofinal = rhoconstruct(R1s, x, 0) + rhoconstruct(R1p, x, 1) + rhoconstruct(R1d, x, 2) +  rhoconstruct(R2s, x, 0) + rhoconstruct(R2p, x, 1) + rhoconstruct(R1f, x, 3)
         
    rhoinitial = a* rhofinal+ (1-a)*rhomix
    Eold = Enew
    Enew = E
    j +=1
    print("Next cycle")
    V0plot, diplot, explot, coplot = drawvec(rhomix)
    
    plt.plot(x[2:Nx], V0plot[2:Nx], linewidth=0.9, label="Total")
    
plt.show()
    
#rhomix = a*rhonew + (1-a) * rhold 
rhomix = rhofinal

Ir = np.trapz(rhomix*x**2*4*m.pi, x )
rhomix = rhomix/Ir *Ne

rhopolmix = np.zeros(Nx)

for i in range (int(Rc/dx), int(Nx)):
    rhopolmix[i] = rhomix[i]
    
N = np.trapz(rhomix*x**2*4*m.pi, x )

deltaN = np.trapz(rhopolmix* x**2* 4*m.pi, x)

alfa = Rc**3*(1+ deltaN/N) /N

# # WF figure
fig = plt.figure()
plt.plot(x[1:int(L/dx)], u1s[1:int(L/dx)], linewidth=0.9, label="1s")
plt.plot(x[1:int(L/dx)], u1p[1:int(L/dx)], linewidth=0.9, label ="1p")
plt.plot(x[1:int(L/dx)], u1d[1:int(L/dx)], linewidth=0.9, label="1d")
plt.plot(x[1:int(L/dx)], u2s[1:int(L/dx)], linewidth=0.9, label="2s")
plt.plot(x[0:int(L/dx)], u1f[0:int(L/dx)], linewidth=0.9, label="1f")
plt.plot(x[0:int(L/dx)], u2p[0:int(L/dx)], linewidth=0.9, label="2p")
plt.legend()
plt.title("Wavefunctions")
#plt.axvline(Rc, color = "grey")
plt.savefig("WF_19_el.png", dpi=300)

# # Density figure

fig1 = plt.figure()
plt.plot(x[1:Nx], rhomix[1:Nx]/rhob, linewidth=0.9)
plt.title("Density of Na with {} electrons".format(Ne))
plt.axhline(1, xmin = 0, xmax = Rc/max(x), color="grey", linestyle="-.", linewidth=0.9)
plt.axvline(Rc, ymin = 0, ymax = 1/max(rhomix/rhob), color="grey", linestyle="-.", linewidth=0.9)
plt.ylabel('$\\rho(r)/\\rho_{0}$')
plt.xlabel('r')
plt.savefig("rho_Na8_el.png", dpi=300)

# # Potential

# V0, di, ex, co = drawvec(rhomix)

# fig2 = plt.figure()
# plt.plot(x[2:Nx], V0[2:Nx], linewidth=0.9, label="Total")
# plt.axhline(E[0,0] , color="blue", linestyle="-.", label="1s", linewidth=0.9)
# plt.axhline(E[0,1] , color="blue", linestyle="-.", label="2s", linewidth=0.9)
# #plt.axhline(E[1,1] , color="blue", linestyle="-.", label="2p", linewidth=0.9)
# plt.plot(x[2:Nx], di[2:Nx], linewidth=0.9, color="magenta", label="direct")
# plt.plot(x[2:Nx], ex[2:Nx] , linewidth=0.9, color="green", label="exchange")
# plt.plot(x[2:Nx], co[2:Nx] , linewidth=0.9, color="red", label="correlation")
# plt.title("Potential")
# # plt.axhline(En1s , color="green", linestyle="-.", label="1s", linewidth=0.9)
# # plt.axhline(En1p , color="blue", linestyle="-.", label="1p", linewidth=0.9)
# # plt.axhline(En1d , color="red", linestyle="-.", label="1d", linewidth=0.9)
# # plt.axhline(En2s , color="cyan", linestyle="-.", label="2s", linewidth=0.9)
# plt.legend()
# plt.savefig("Pot_20_el.png", dpi=300)


AppKit.NSBeep()
