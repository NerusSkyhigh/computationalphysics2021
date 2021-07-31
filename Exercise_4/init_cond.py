# QUANTUM MONTE CARLO

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt

# Number of particles 

N_p = 32 

# set the value of the density given on the paper

rho = 21.86 # particles per cubic nanometer

# Since each cell hosts 4 particles in the initial configuration,
# we just have 8 cells as the text mentions. Then, the volume of the whole cube
# is (2a)^3, where a is the cell side. So we can find the cell side as

a = 0.5 * (N_p / rho)**(1.0/3.0) # Cell side in nanometers 

# We can now place the particles as we desire in an FCC lattice
# To store the particle positions, we use a 32x3 numpy array 

particles = np.zeros((32,3))

# Let us define a function to make the initial conditions

def initcond(part):
    
    # Particles per side. In this case it's super easy, we already know it's 2
    
    ppl = 2
    
    pind = 0 

    # iterate over box sides,
    
    for i in range(0,ppl):
        
        trslx = a * i
        
        for j in range(0,ppl):
            
            trsly = a * j
            
            for k in range(0, ppl): 
                
                trslz = a * k 

                particles[pind, :] = np.array([trslx, trsly, trslz])
                particles[pind+1, :] = np.array([trslx+a/2, trsly+a/2, trslz])
                particles[pind+2, :] = np.array([trslx, trsly+a/2, trslz+a/2])
                particles[pind+3, :] = np.array([trslx+a/2, trsly, trslz+a/2])

                pind += 4
    
    return part;


# This function does the same thing in a way that is compatible with the optimisation provided by njit

def initcond_1(part):
    
    pind = 0
    
    for k in range(0,4):
        
        zval = k * a / 2 # set z coordinate
    
        for i in range(0,4):
            
            xval = i * a / 2 # set x coordinate
            
            for j in range(0,2):
            
                if k%2 == 0:
            
                    yval = j * a + i%2 * a/2 # set y coordinate
            
                else:
                    
                    yval = (j + 1/2) * a - i%2 * a/2 # set y coordinate
        
                part[pind, 0] = xval 
                part[pind, 1] = yval
                part[pind, 2] = zval
                
                pind += 1
    
    return part;

particles = initcond(particles)

particles1 = initcond(particles)

# Let us verify our success with a plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles1[:,0], particles1[:,1], particles1[:,2])

