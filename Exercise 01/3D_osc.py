# Harmonic Oscillator
import time
import numpy as np
from numba import njit

Nx = 5000
dE = 0.00001
LMAX = 3
L = 8.5
dx=L/Nx

x = np.arange(0,dx*Nx,dx)

@njit
def potential(x, l):
    
    if x == 0:
        V = 0
    else:
        V = 0.5 * x**2 + (l *(l + 1)) / (2.0 * x**2)
    
    return V

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

@njit # Add njit to enjoy that wonderful speedup of compiled programs
def iterating(lmax):
    
    psi = np.zeros(Nx)
    lastnow = 1.0
    restart = 1.0
    
    energies = np.zeros((3, lmax))
    
    for l in range(0, lmax, 1):
         
        psi[0] = 0.0
        psi[1] = pow(dx, l + 1)
        
        if l == 0:
            Ener = 0.0
        else: 
            Ener = restart
            
        for n in range(l, l + 3, 1):
            
            rat = 1.0 # Reset rat!!
            
            while rat > 0:
                
                for i in range(2, Nx, 1):
                    
                    psi[i] = numerov(psi[i-2], psi[i-1], i, Ener, l)
        
                lastbefore = lastnow 
                lastnow = psi[Nx-1]
                rat = lastnow / lastbefore
                
                Ener = Ener + dE
            
            zero = Ener- dE - (dE / (lastnow - lastbefore) ) * lastnow
            
            energies[n-l,l] = zero
            
            if n == (l + 1):
                restart = zero
            
    return energies
         
    
energies = np.zeros(LMAX)

begin = time.time()

energies = iterating(LMAX)
    
end = time.time()

print("This took me "+str(end-begin)+" seconds.") 
    