# Harmonic Oscillator in 3D

import numpy as np
import matplotlib.pyplot as plt

Nx = 8000
dx = 0.001
dE = 0.00001
nmax = 3

x = np.arange(0,dx*Nx,dx)

#Function which computes the potential

def potential(r, l):
    
    if r == 0:
        V = 0
    else:
        V = 0.5 * r**2 + (l * (l+1)) / (2.0 * r**2)
    
    return V

def numerov(psi1, psi2, i, En, l):
    
    p1 = potential(dx * (i-2), l)
    p2 = potential(dx * (i-1), l)
    pf = potential(dx * i, l)
    
    k1 = 2.0 * (En - p1)
    k2 = 2.0 * (En - p2)
    kf = 2.0 * (En - pf)

    num1 =  psi2 * (2.0 - (5.0 / 6.0) * k2 * dx**2) 
    num2 = psi1 * (1.0 + (1.0 / 12.0) * k1 * dx**2)
    den = 1.0 + (1.0 / 12.0) * kf * dx**2

    psif = (num1 - num2) / den
    
    return psif


En = 11.5

n = 4
l = 2

psi = np.zeros(Nx) 
psilow = np.zeros(Nx) 
psihigh = np.zeros(Nx) 
            
psi[0] = 0
psi[1] = pow(dx, l + 1)  

psilow[0] = 0
psilow[1] = pow(dx, l + 1)  

psihigh[0] = 0
psihigh[1] = pow(dx, l + 1)  
        
# Compute wavefunction
        
for i in range(2,Nx,1):
    psi[i] = numerov(psi[i-2], psi[i-1], i, En, l)
    psilow[i] = numerov(psilow[i-2], psilow[i-1], i, En-dE, l)
    psihigh[i] = numerov(psihigh[i-2], psihigh[i-1], i, En+dE, l)
    
plt.title("Wavefunction n="+str(n)+", l="+str(l))
plt.xlabel("r")
plt.ylabel("R(r)")
plt.plot(x,psi, label='E = '+str(2*n + l + 1.5))
plt.plot(x,psilow, label='E = '+str(2*n + l + 1.5)+' - dE')
plt.plot(x,psihigh, label='E = '+str(2*n + l + 1.5)+' +dE')
plt.xlim([0,dx*Nx])
plt.ylim([-0.25,0.4]) 
plt.legend()
    
    