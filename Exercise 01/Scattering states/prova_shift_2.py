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
L = 10.0 # Mesh length
deltax = L / Nx # mesh spacing
#fac = 0.03528
fac = 0.03528
rmax = 5.0


x = np.arange(0.5,deltax*Nx,deltax)
E = np.arange(0.01, 0.59, dE)

#POTENTIAL FUNCTION

@njit
def potential(r, a):

    
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
    phsh = m.atan( (num1 - num2) / (den1 - den2) ) ;

    return phsh


# FUNCTION FOR THE CROSS SECTION


@njit
def main():
    
    #Vectors for psi and cross section results
    psi = np.zeros(Nx)
    cross = np.zeros(len(E))
    phasematr = np.zeros((len(E), lmax))
    shi = np.zeros((len(E), lmax))
    lam = np.zeros(len(E))
 
    j = 0
    for En in E:

            
        psi[500] = m.exp(-(1.16 / 0.5)**5 ) #we don't start from zero but from 0.5
        psi[501] = m.exp(-(1.16/ (0.5+deltax))**5 ) 


        for l in range(0, lmax, 1):
        
            for i in range(502,Nx,1):
            
                psi[i] = numerov(psi[i-2], psi[i-1],i, En, l);

            lam = (2*m.pi)/(m.sqrt(En / fac))
            N1 = int(5/ deltax) 
            N2 = N1 + int(lam/(4*deltax))
            
        
            shift = phaseshift(N1, N2, l, psi[N1], psi[N2], En);
        
            phasematr[j,l] = pow(m.sin(shift),2) * (2 * l + 1) * ( (4 * m.pi) / (En / fac) )
            
            cross[j] += pow(m.sin(shift),2) * (2 * l + 1) * ( (4 * m.pi) / (En / fac) )
           
            shi[j,l]= shift            
            l +=1
            
        j +=1

    return cross, phasematr, shi


start = time.time()
cross, phasematrix, shi = main()
cross = (3.18)**2 * cross
end = time.time()

print("The calculation took {} seconds".format(end-start))



shift_required = shi[290,:];

E = E * 5.9
fig1 = plt.figure()
plt.plot(E, shi[:,0], linewidth = 1, color='green')
plt.plot(E, shi[:,1], linewidth = 1, color='green')
plt.plot(E, shi[:,2], linewidth = 1, color='green')
plt.plot(E, shi[:,3], linewidth = 1, color='green')
plt.plot(E, shi[:,4], linewidth = 1, color='black')
plt.plot(E, shi[:,5], linewidth = 1, color='red')
plt.plot(E, shi[:,6], linewidth = 1, color='blue')
plt.title("Phase shifts")
plt.xlabel("E")
plt.ylabel(r"$\phi_l$")
plt.savefig('shifts.png', dpi=600)

fig2 = plt.figure()
plt.plot(E, phasematrix[:,0], linewidth = 1, color='green')
plt.plot(E, phasematrix[:,1], linewidth = 1, color='green')
plt.plot(E, phasematrix[:,2], linewidth = 1, color='green')
plt.plot(E, phasematrix[:,3], linewidth = 1, color='green')
plt.plot(E, phasematrix[:,4], linewidth = 1, color='black')
plt.plot(E, phasematrix[:,5], linewidth = 1, color='red')
plt.plot(E, phasematrix[:,6], linewidth = 1, color='blue')
plt.title("Cross sections for each value of l")
plt.xlabel("E")
plt.ylabel(r"$\sigma_l$")
plt.savefig('singlecross.png', dpi=600)

fig3 = plt.figure()
plt.plot(E, cross, linewidth = 1, color='black')
plt.title("Cross Section")
plt.xlabel("E")
plt.ylabel(r"$\sigma_{tot}$")
plt.savefig('crossec.png', dpi=600)
