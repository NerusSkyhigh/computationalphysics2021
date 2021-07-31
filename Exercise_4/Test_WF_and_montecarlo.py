# QUANTUM MONTE CARLO

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import random as rd

# Number of particles 

N_p = 32 

# Number of equilibration and simulation steps

N_eq = 1000
N_steps = 500000

# value of the hbar/2m constant

hbm = 9.091*1e-4

# set the value of the density given on the paper

rho = 21.86 # particles per cubic nanometer

# Since each cell hosts 4 particles in the initial configuration,
# we just have 8 cells as the text mentions. Then, the volume of the whole cube
# is (2a)^3, where a is the cell side. So we can find the cell side as

a = 0.5 * (N_p / rho)**(1.0/3.0) # Cell side in nanometers 

sigma = 0.2556

a = a / sigma # cell side in sigma units

eps = 10.22 # Energy unit, in kelvin...i do not think it is needed

# b parameter... we'll work on this as a variational parameter later

b = 1

# Choice of the Delta parameter for variations

Delta = 0.001

# We can now place the particles as we desire in an FCC lattice
# To store the particle positions, we use a 32x3 numpy array 

particles = np.zeros((32,3))

# FUNCTION TO SET INITIAL CONDITIONS 

# Sets the x,y,z coordinates of all 32 particles on a fcc lattice in a way that
# is compatible with the njit decorator (may be useless, it's already fast 
# enough in the unjittable version)

@njit
def initcond(part):
    
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


# FUNCTION TO COMPUTE WF 


@njit
def WF(part, b):
    
    wf = 1
    
    for i in range(0, N_p):
        
        for j in range(0, i):
            
            # compute the distance between two particles 
            
            dx = part[i, 0] - part[j, 0]
            dx = dx - 2*a* m.floor(dx / (2*a)) # PBC
            dy = part[i, 1] - part[j, 1]
            dy = dy - 2*a* m.floor(dy / (2*a)) # PBC
            dz = part[i, 2] - part[j, 2]
            dz = dz - 2*a* m.floor(dz / (2*a)) # PBC 
            
            rij = m.sqrt(dx**2 + dy**2 + dz**2)
            
            wf = wf * m.exp( -0.5 * ( b / rij )**5 )
    
    return wf 

# FUNCTION FOR THE METROPOLIS ALGORITHM

@njit
def metropolis(part, N_acc, wf2_old):
    
    # First, we need to build a new configuration
    # We start from the initial one
    
    new_part = part
    
    # Variation of positions with PBC
    
    for ind in range(0,N_p):
        
        for co in range(0, 3): # Loop over x,y,z particle coordinates
        
            rand1 = rd.uniform(0,1) # Pick random number
            new_part[ind, co] = part[ind, co] - Delta * (rand1 - 0.5) # Set coordinate
            new_part[ind, co] = new_part[ind, co] - 2*a * m.floor( new_part[ind, co] / (2*a) ) # PBC
        
    # Now, we must evaluate the acceptance ratio using the ratio of the square moduli of the WF
    
    wf_new = WF(new_part, b)
    
    wf2_new = wf_new**2
    
    acc = wf2_new / wf2_old 
    
    # Decide acceptance
    
    xi = rd.uniform(0,1)
    
    wf2 = wf2_old
    
    if xi < acc: # new_part becomes part, we accept the MC move
        
        part = new_part
        
        N_acc += 1 # Increase the number of accepted moves
        
        wf2 = wf2_new
        
        
    return part, N_acc, wf2


# FUNCTION TO PERFORM THE ITERATIVE PROCEDURE

@njit
def montecarlo(part):
    
    # First, the equilibration steps 
    
    # We also want to store the WF square modulus in a vector, since it is the 
    # probability density we sample, apart from a normalisation (is it relevant?)
    # So we can check what is going on with it... 
    # What do we expect the WF to be? do we expect it to be sth interesting? 
    
    pd = np.zeros(N_eq+N_steps)
    
    wf2 = WF(part,b)
    
    N_acc1 = 0
    
    for s in range(0, N_eq):
        
        part, N_acc1, wf2 = metropolis(part, N_acc1, wf2) 
        
        pd[s] = wf2
    
    # Then, the simulation steps
    
    N_acc2 = 0

    for s in range(0, N_steps):
        
        part, N_acc2, wf2 = metropolis(part, N_acc2, wf2) 
        
        pd[s+N_eq] = wf2
        
    accf = N_acc2/N_steps
        
    return part, pd, accf


# EXECUTION OF THE PROCEDURE

# Start procedure timing

start = time.time()

# Set initial conditions

particles = initcond(particles)

# Let us verify our success with a plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2])
plt.title("Initial Conditions")

# Run the MC simulation

particles, pd, accf = montecarlo(particles)

print("\n The fraction of accepted moves was {}".format(accf))

# Verify graphically that the simulation actually changed where the particles are
# Why? because if it does not do so then it surely does not work 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2])
plt.title("New Configuration")

# Let us draw the square modulus of the WF at each step 

steps = np.arange(0,N_eq+N_steps)

fig = plt.figure()
plt.plot(steps, pd)
plt.title("Probability Density")

# End procedure timing 

end = time.time()

print("\n The simulation took me {} seconds".format(end-start))

