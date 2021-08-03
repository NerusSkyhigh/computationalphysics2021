# QUANTUM MONTE CARLO

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import random as rd

N_steps = 50000 # Simulation steps 

N_p = 32 # Particles

particles = np.zeros((N_p,3)) # Array to store the particles 

rho = 21.86 # particles per cubic nanometer

a = 0.5 * (N_p / rho)**(1.0/3.0) / 0.2556 # cell side in units of sigma

# FUNCTION TO SET INITIAL CONDITIONS ON AN FCC LATTICE 

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

# COMPUTE THE WAVEFUNCTION'S SQUARE MODULUS

@njit
def WF(part, b):
    
    wf2 = 0.0
    expon = 0.0
    
    for i in range(0,N_p): #Loop over first particle
        for j in range(0, i): #Loop over other particles
            
            # Compute distances along the axes
            dx = part[i,0] - part[j,0]
            dx = dx - 2*a*m.floor(dx / (2*a))
            dy = part[i,1] - part[j,1]
            dy = dy - 2*a*m.floor(dy / (2*a))
            dz = part[i,2] - part[j,2]
            dz = dz - 2*a*m.floor(dz / (2*a))
            
            # Compute distance between barticles
            
            rij = m.sqrt( dx**2 + dy**2 + dz**2 )
            
            expon += - (b / rij)**5 # we multiply it by 2 to directly evaluate the square
                
    wf2 = m.exp(expon)
    
    return wf2

# METROPOLIS ALGORITHM FUNCTION 

@njit
def metropolis(part, wf2_old, N_acc, b, Delta):
    
    new_part = np.zeros((N_p,3)) # new set of particles
    
    for i in range(0, N_p):
        for co in range(0,3):
        
            # Random displacement of each coordinate of each particle
            new_part[i,co] = part[i,co] - Delta * (rd.uniform(0,1) - 0.5)
            # PBC
            # new_part[i,co] = new_part[i,co] - 2*a * m.floor( new_part[i,co] / (2*a))
            
    # Compute acceptance ratio
    
    wf2_new = WF(new_part, b)
    acc = wf2_new / wf2_old
    
    # Decide acceptance 
    
    xi = rd.uniform(0,1)
    
    if xi < acc:
        
        wf2_old = wf2_new
        
        part = new_part
        
        N_acc +=1
        
    return part, wf2_old, N_acc
        
# RUN METROPOLIS ITERATIVELY

@njit
def montecarlo(part, b):
    
    part = initcond(part) # Sets the initial conditions for this MC run
    
    Delta = 0.07
    
    # First, let us perform an equilibration phase
    N_eq = 5000
    
    N_acc1 = 0
    wf2 = WF(part, b)
    
    for s in range(0,N_eq):
        
        part, wf2, N_acc1 = metropolis(part, wf2, N_acc1, b, Delta)
        
        # An adaptive scheme for Delta 
          
        if s != 0  and s%5 == 0:
          
            if N_acc1 / s >= 0.55:
              
                Delta += 0.05 * 0.07
             
            elif N_acc1 / s <= 0.45:
             
                Delta -= 0.05 * 0.07
        
    # Then, the actual simulation
    
    N_s = 50000
    
    N_acc2 = 0
    
    for s in range(0,N_s):
        
        part, wf2, N_acc2 = metropolis(part, wf2, N_acc2, b, Delta)
        
        # An adaptive scheme for Delta 
          
        if s != 0  and s%5 == 0:
          
            if N_acc2 / s >= 0.55:
              
                Delta += 0.05 * 0.07
             
            elif N_acc2 / s <= 0.45:
             
                Delta -= 0.05 * 0.07
                
    frac = N_acc2 / N_s
    
    return part, frac
    
# MC SIMULATION

# start timing

start = time.time()


# Let us draw the initial conditions

in_particles = initcond(particles)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(in_particles[:,0], in_particles[:,1], in_particles[:,2], marker=".")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Initial Conditions")

# MC simulation

particles, acc = montecarlo(particles, 1)

# Let us draw the final configuration

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2], marker=".")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("New Configuration")

# Print some results 

print("\n The fraction of accepted moves was {}".format(acc))

# End the timing

end = time.time()

print("The simulation took me {} seconds".format(end-start))

