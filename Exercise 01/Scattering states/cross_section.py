# Harmonic Oscillator

import numpy as np
import math as m
from numba import njit
import matplotlib.pyplot as plt
import time

# Setting some initial parameters 

Nx = 14000 #number of mesh points
dE = 0.0001 # energy step
lmax = 6 # Maximum value of l
L = 20.0 # Mesh length
deltax = L / Nx # mesh spacing
fac = 0.03528

x = np.arange(0,deltax*Nx,deltax)
E = np.arange(dE, 0.549, dE)

#POTENTIAL FUNCTION

@njit
def potential(r, a):

    if r == 0:

      r = deltax;

    t1 = 1.0 / pow(r, 12);
    t2 = 1.0 / pow(r, 6);
    lenn = t1 - t2;
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
def j(l, x):
    if x == 0:
        x = 1E-30
        
    if l == -1:
        return np.cos(x) / x
    elif l == 0:
        return np.sin(x) / x
    else:
        return (2*(l-1)+1)/x * j(l-1, x) - j(l-2, x)
    
    
@njit   
def n(l, x):
    if x == 0:
        x = 1E-30
        
    if l == -1:
        return np.sin(x) / x
    elif l == 0:
        return -1*np.cos(x) / x
    else:
        return (2*(l-1)+1)/x*n(l-1, x) - n(l-2, x)
    
#PHASE SHIFT FUNCITON

@njit
def phaseshift(n1, n2, l, psi1, psi2, Energy):

    Kbig = ( psi1 * (deltax * n2) ) / ( psi2 * (deltax * n1) );
    Ksmall = m.sqrt(Energy / fac);

    num1 = Kbig * j(l, Ksmall * (deltax * n2));
    num2 = j(l, Ksmall * (deltax * n1) );
    den1 = Kbig * n(l, Ksmall * (deltax * n2));
    den2 = n(l, Ksmall * (deltax * n1) );
    phsh = m.atan( (num1 - num2) / (den1 - den2) );

    return phsh


# CODE CORE

@njit
def main():
    
    #Vectors for psi and cross section results
    psi = np.zeros(Nx)
    cross = np.zeros(len(E))
    j=0
    
    for En in E:
        
        for l in range(0,lmax,1):
            
            psi[0] = m.exp(-(1.1632127145 / 0.5)**5 )
            psi[1] = m.exp(-(1.1632127145 / (0.5+deltax))**5 )
            
            for i in range(2,Nx,1):
            
                psi[i] = numerov(psi[i-2], psi[i-1], i + int(0.5/deltax), En, l);
  
            N1 = 1700
            N2 = 1720

            shift = phaseshift(N1, N2, l, psi[N1], psi[N2], En);

            # print(shift)
            
            cross[j] += pow(m.sin(shift),2) * (2 * l + 1) * ( (4 * m.pi) / (En / fac) );

        j +=1
        En += dE
        
    return cross

start = time.time()
crossection = main()
end = time.time()

print("The calculation took {} seconds".format(end-start))

fig = plt.figure()
plt.plot(E, crossection, linewidth = 1, color='black')
plt.title("Cross Section")
plt.xlabel("E")
plt.ylabel(r"$\sigma_{tot}$")
plt.savefig('crossec.png', dpi=600)

