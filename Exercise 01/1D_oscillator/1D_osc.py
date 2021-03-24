# Harmonic Oscillator

import numpy as np
import matplotlib.pyplot as plt

Nx = 5000
dx = 0.001
dE = 0.00001
nmax = 5

x = np.arange(0,dx*Nx,dx)

def potential(x):
    
    V = 0.5 * x**2
    
    return V

def numerov(psi1, psi2, i, En):
    
    p1 = potential(dx * (i-2))
    p2 = potential(dx * (i-1))
    pf = potential(dx * i)
    
    k1 = 2.0 * (En - p1)
    k2 = 2.0 * (En - p2)
    kf = 2.0 * (En - pf)

    num1 =  psi2 * (2.0 - (5.0 / 6.0) * k2 * dx**2) 
    num2 = psi1 * (1.0 + (1.0 / 12.0) * k1 * dx**2)
    den = 1.0 + (1.0 / 12.0) * kf * dx**2

    psif = (num1 - num2) / den
    
    return psif


En = 0.5
lastnow = 1.0
energies = np.zeros(nmax)

for n in range(0,nmax,1):
        
    j = 0
    prod = 1.0
        
    while prod >=0:
    
        # Initial conditions
                
        psi = np.zeros(Nx) 
        
        if n%2 == 1:
            psi[0] = 0
            psi[1] = dx
        elif n == 0 or n == 4:
            psi[0] = 1.0
            psi[1] = 1.0
        elif n == 2:
            psi[0] = -1
            psi[1] = -1 
    
        # Compute wavefunction
    
        for i in range(2,Nx,1):
            psi[i] = numerov(psi[i-2], psi[i-1], i, En)
                 
        j += 1
            
        if j == 0:
            lastbefore = psi[Nx-1]
        else: 
            lastbefore = lastnow
                
        lastnow = psi[Nx-1]
        prod = lastbefore * lastnow
        print(En)
            
        En += dE
        
    energies[n] = En - (dE / (lastnow - lastbefore) ) * lastnow
    print("Ho trovato il livello con energia "+str(energies[n])+" !!!")
    

pot = [] 
en = np.zeros(Nx)
 
for j in range(0,Nx*dx,dx):
    pot[j] = potential(dx*j)

fig = plt.figure()
plt.plot(x, pot)

for k in range(0,nmax,1):
    plt.axhline(energies[k],0,dx*Nx)

plt.xlim([0,dx*Nx])
plt.ylim([-1,9]) 
    
    
    
    
    