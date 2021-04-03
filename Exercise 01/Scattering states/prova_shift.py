# Harmonic Oscillator

import numpy as np
import math as m
from numba import njit
import matplotlib.pyplot as plt
import time

# Setting some initial parameters 

Nx = 10000 #number of mesh points
dE = 0.001 # energy step
lmax = 7 # Maximum value of l
L = 30.0 # Mesh length
deltax = L / Nx # mesh spacing
#fac = 0.03528
fac = 0.02799
rmax = 5.0
Emin = dE

x = np.arange(0,deltax*Nx,deltax)
E = np.arange(0.02, 0.59, dE)

#POTENTIAL FUNCTION

@njit
def potential(r, a):

    if r == 0:

         r = 1E-30;
    
    t1 = 1.0 / pow(r, 12);
    t2 = 1.0 / pow(r, 6);
    lenn = t1 - t2;
    
    if r > rmax:
    
        lenn = 0
        
    p = 4.0 * lenn + fac * (a * (a + 1) ) / (r * r);

    return p


# NUMEROV FUNCTION

@njit
def numerov(psi1,psi2, ind, En, ang):

    p1 = potential(deltax * (ind - 2), ang);
    p2 = potential(deltax * (ind - 1), ang);
    pf = potential(deltax * ind, ang);

    k1 = (1. / fac) * (En - p1);
    k2 = (1. / fac) * (En - p2);
    kf = (1. / fac) * (En - pf);

    valn1 = psi2 * (2 - (5.0 / 6.0) * deltax * deltax * k2);
    valn2 = psi1 * (1 + (1.0 / 12.0) * deltax * deltax * k1);
    vald = 1 + (1. / 12.) * deltax * deltax * kf;

    f = (valn1 - valn2) / vald;

    return f


#BESSEL AND NEUMANN FUNCTION

@njit
def bessj(l, x):
    if x == 0:
        x = 1E-30
        
    if l == -1:
        return np.cos(x) / x
    elif l == 0:
        return np.sin(x) / x
    else:
        return (2*(l-1)+1)/x * bessj(l-1, x) - bessj(l-2, x)
    
    
@njit   
def bessn(l, x):
    if x == 0:
        x = 1E-30
        
    if l == -1:
        return np.sin(x) / x
    elif l == 0:
        return -1*np.cos(x) / x
    else:
        return (2*(l-1)+1)/x*bessn(l-1, x) - bessn(l-2, x)
    
#PHASE SHIFT FUNCITON


@njit
def phaseshift(n1, n2, l, psi1, psi2, Energy):

    Kbig = ( psi1 * (deltax * n2) ) / ( psi2 * (deltax * n1) );
    Ksmall = m.sqrt(Energy / fac);

    num1 = Kbig * bessj(l, Ksmall * (deltax * n2));
    num2 = bessj(l, Ksmall * (deltax * n1) );
    den1 = Kbig * bessn(l, Ksmall * (deltax * n2));
    den2 = bessn(l, Ksmall * (deltax * n1) );
    phsh = m.atan( (num1 - num2) / (den1 - den2) );

    return phsh


# FUNCTION FOR THE CROSS SECTION


@njit
def main():
    
    #Vectors for psi and cross section results
    psi = np.zeros(Nx)
    cross = np.zeros(len(E))
    phasematr = np.zeros((lmax, len(E)))
    
    l = 3
    j = 0
        
    for En in E:
    
        psi[0] = m.exp(-(1.19045 / 0.5)**5 )
        psi[1] = m.exp(-(1.19045 / (0.5+deltax))**5 ) 
        
        for l in range(0, lmax, 1):
        
            for i in range(2,Nx,1):
            
                psi[i] = numerov(psi[i-2], psi[i-1], i + int(0.5/deltax), En, l);
          
            N1 = int(15/ deltax)
            N2 = N1 + 100
        
            shift = phaseshift(N1, N2, l, psi[N1], psi[N2], En);
        
            phasematr[l,j] = pow(m.sin(shift),2) * (2 * l + 1) * ( (4 * m.pi) / (En / fac) )
            
            cross[j] += pow(m.sin(shift),2) * (2 * l + 1) * ( (4 * m.pi) / (En / fac) )
    
        
        j +=1
        
    return cross, phasematr


start = time.time()
crossection, phasematrix = main()
crossection = (3.18)**2 * crossection
end = time.time()

print("The calculation took {} seconds".format(end-start))

fig = plt.figure()
plt.plot(E, phasematrix[0,:]+phasematrix[1,:]+phasematrix[2,:]+phasematrix[3,:], linewidth = 1, color='green')
plt.plot(E, phasematrix[4,:], linewidth = 1, color='black')
plt.plot(E, phasematrix[5,:], linewidth = 1, color='red')
plt.plot(E, phasematrix[6,:], linewidth = 1, color='blue')
plt.title("Phase Shifts")
plt.xlabel("E")
plt.ylabel(r"$\phi_l$")
plt.savefig('shifts.png', dpi=600)

fig2 = plt.figure()
plt.plot(E, crossection, linewidth = 1, color='black')
plt.title("Cross Section")
plt.xlabel("E")
plt.ylabel(r"$\sigma_{tot}$")
plt.savefig('crossec.png', dpi=600)
