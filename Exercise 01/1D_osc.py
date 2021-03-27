# Harmonic Oscillator
import time
import numpy as np
from numba import njit


Nx = 3000
dE = 0.00001
NMAX = 5
L = 6
dx=L/Nx

x = np.arange(0,dx*Nx,dx)

@njit
def potential(x):
    
    V = 0.5 * x**2
    
    return V

@njit
def numerov(psi1, psi2, i, E):
    
    p1 = potential(dx * (i-2))
    p2 = potential(dx * (i-1))
    pf = potential(dx * i)
    
    k1 = 2.0 * (E - p1)
    k2 = 2.0 * (E - p2)
    kf = 2.0 * (E - pf)

    num1 =  psi2 * (2.0 - (5.0 / 6.0) * k2 * dx**2) 
    num2 = psi1 * (1.0 + (1.0 / 12.0) * k1 * dx**2)
    den = 1.0 + (1.0 / 12.0) * kf * dx**2

    psif = (num1 - num2) / den
    
    return psif

@njit
def iterating(Eninit, energyvec, nmax):
    
     for n in range(0,nmax,1):
        
        rat = 1.0
        lastnow = 1.0
            
        while rat >=0:
        
            # Initial conditions
                    
            psi = np.zeros(Nx) 
            
            if n == 0 :
                psi[0] = 1.0
                psi[1] = 1.0
            elif n == 1:
                psi[0] = 0.0
                psi[1] = dx
            elif n == 2:
                psi[0] = -1.0
                psi[1] = -1.0 
            elif n == 3:
                psi[0] = 0.0
                psi[1] = -dx 
            elif n == 4:
                psi[0] = 1.0
                psi[1] = 1.0
        
            # Compute wavefunction
        
            for i in range(2,Nx,1):
                psi[i] = numerov(psi[i-2], psi[i-1], i, Eninit)
                     
            lastbefore = lastnow
            lastnow = psi[Nx-1]
            rat = lastbefore / lastnow
            #print(En, rat)
                
            Eninit += dE
            
        energyvec[n] = Eninit-dE - (dE / (lastnow - lastbefore) ) * lastnow

     return energyvec

En = 0.0
energies = np.zeros(NMAX)

begin = time.time()

energies = iterating(En, energies, NMAX)
    
end = time.time()

print("This took me "+str(end-begin)+" seconds.") 
    


    