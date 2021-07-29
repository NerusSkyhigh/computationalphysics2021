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

N_eq = 100

N_steps = 10000 

# value of the hbar/2m constant

hbm = 9.091*1e-4

# set the value of the density given on the paper

rho = 21.86 # particles per cubic nanometer

# Since each cell hosts 4 particles in the initial configuration,
# we just have 8 cells as the text mentions. Then, the volume of the whole cube
# is (2a)^3, where a is the cell side. So we can find the cell side as

a = 0.4 * (N_p / rho)**(1.0/3.0) # Cell side in nanometers 

sigma = 0.2556

a = a / sigma # cell side in sigma units

# b parameter... we'll work on this as a variational parameter later

b = sigma

# Choice of the Delta parameter for variations

Delta = 0.05 * a 

# We can now place the particles as we desire in an FCC lattice
# To store the particle positions, we use a 32x3 numpy array 

particles = np.zeros((32,3))

# FUNCTION TO MAKE INITIAL CONDITIONS

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


# FUNCTION TO COMPUTE WF SQUARE MODULUS 

# This has a "problem"... of course when (b/r) is big (small r or big b)
# The WF goes to 0 almost immediately and we can not divide by 0 in the metropolis
# algorithm when we compute the acceptance ratio... am i missing sth either here
# or in the metropolis algorithm?

def WF(part, b):
    
    wf = 1
    
    for i in range(0, N_p):
        
        for j in range(0, i):
            
            # compute the distance between two particles 
            
            dx = part[i, 0] - part[j, 0]
            dy = part[i, 1] - part[j, 1]
            dz = part[i, 2] - part[j, 2]
            
            rij = m.sqrt(dx**2 + dy**2 + dz**2)
            
            # I use the next step to correct the fact that divergences make the WF go to 0.... 
            # but how should it be done??
            # Are there other mistakes i am unaware of? Very likely
            
            if rij < 0.5 * sigma:
            
                rij = 0.5 * sigma
                
            # Compute the wavefunction
            
            wf = wf * m.exp( -0.5 * ( b / rij )**5 )
            
            #print(wf)
    
    return wf 

# FUNCTION FOR THE METROPOLIS ALGORITHM

def metropolis(part, N_acc):
    
    # First, we need to build a new configuration
    # We start from the initial one
    
    new_part = part
    
    # Old WF
    
    wf_old = WF(part, b)
    
    #print("WF OLD IS {}".format(wf_old))
    
    wf2_old = wf_old**2
    
    # Variation of positions with PBC
    
    for ind in range(0,N_p):
        
        for co in range(0, 2): # Loop over x,y,z particle coordinates
        
            rand1 = rd.uniform(0,1) # Pick random number
            new_part[ind, co] = part[ind, co] - Delta * (rand1 - 0.5) # Set coordinate
            new_part[ind, co] = new_part[ind, co] - 2*a * m.floor( new_part[ind, co] / (2*a) ) # PBC
        
    # Now, we must evaluate the acceptance ratio using the ratio of the square moduli of the WF
    
    wf_new = WF(new_part, b)
    
    wf2_new = wf_new**2
    
    acc = wf2_new / wf2_old 
    
    # Decide acceptance
    
    xi = rd.uniform(0,1)
    
    if xi < acc: # new_part becomes part, we accept the MC move
        
        part = new_part
        
        N_acc += 1 # Increase the number of accepted moves
        
        
    return part, N_acc


# FUNCTION TO COMPUTE THE ENERGY (possibly in both ways)

# verify this because i am not very convinced of what you have done
# Implment the second way of computing energies to check for mistakes

# Ok i irsultati non mi sono piaciuti nemmeno un po' nonononononono

def energy(part):
    
    en = 0
    
    # First let us compute the gradient of the interparticle potential (divided by r)
    # We also compute the laplacian here to save time
    
    # iterate over all particles
    
    dU = np.zeros((N_p,3)) # gradient
    
    d2U = 0 # scalar variable for the laplacian
    
    
    for l in range (0, N_p):
        
        # The gradient is already set to zero, no need to do it here
        
        for i in range(0, N_p):
            
            if i != l: 
                
                dx = part[l, 0] - part[i, 0]
                dx = dx - 2*a * m.floor(dx / (2*a))
                dy = part[l, 1] - part[i, 1]
                dy = dy - 2*a * m.floor(dy / (2*a))
                dz = part[l, 2] - part[i, 2]
                dz = dz - 2*a * m.floor(dy / (2*a))       
                
                rlj = (dx**2 + dy**2 + dz**2)**0.5
            
                der = - 5 * b**5 / rlj**7
                
                dU[l, 0] += der*dx
                dU[l, 1] += der*dy
                dU[l, 2] += der*dz
                
                d2U +=  -4 * der # i might be wrong about this laplacian
                
    # Now we can compute the energy   
    # First, the piece with the scalar prod of the gradient... am i doing it right? 
    
    
    e1 = 0
    
    for ind in range(0, N_p):
        for co in range(0,2):
            e1 += dU[ind, co]**2 
    
    
    # Finally sum up to get the local kinetic energy 
    
    en = -hbm * (e1/4 - d2U / 2)
    
    
    # Now we also have to do the potential energy, right? Let us do it
    
    P_e = 0
    
    for l in range (0, N_p):
       
       for i in range(0, l):
           
           if i != l: 
               
               dx = part[l, 0] - part[i, 0]
               dx = dx - 2*a * m.floor(dx / (2*a))
               dy = part[l, 1] - part[i, 1]
               dy = dy - 2*a * m.floor(dy / (2*a))
               dz = part[l, 2] - part[i, 2]
               dz = dz - 2*a * m.floor(dy / (2*a))       
               
               rlj2 = dx**2 + dy**2 + dz**2
               
               rlj6 = 1 / rlj2**3
               rlj12 = rlj6**2
               
               P_e += 4 * (rlj12 - rlj6)
               
               
    t_e = en + P_e 
    
    return t_e


# FUNCTION TO PERFORM THE ITERATIVE PROCEDURE

# Of course functions to get quantities and evaluate statistical errors / autocorrelations 
# must be included here. So far, the std estimate does not account for the effect due 
# to the autocorrelation... we could work on it

def montecarlo(part):
    
    # First, the equilibration steps 
    
    N_acc = 0
    cumen = 0
    cumen2 = 0
    
    for s in range(0, N_eq):
        
        part, N_acc = metropolis(part, N_acc) 
        
        # Compute local energy 
        
        ener = energy(part)
        
        # Cumulate the energy and its square
        
        cumen += ener
        cumen2 += ener**2
        
    # Compute mean energy and std (we could do this only with the next step, but why not)
        
    mean_energy = cumen / N_eq
    std_energy = m.sqrt( (1 / (N_eq-1) ) * (cumen2 / N_eq - mean_energy**2) )
        
    print("\n La frazione di proposte accettate è: {}".format(N_acc/N_eq))
    print("\n L'energia è : {} pm {}".format(mean_energy, std_energy))
    
    # Then, the simulation steps
    
    N_acc = 0
    cumen = 0
    cumen2 = 0
    
    for s in range(0, N_eq):
        
        part, N_acc = metropolis(part, N_acc) 
        
        # Compute local energy 
        
        ener = energy(part)
        
        # Cumulate the energy and its square
        
        cumen += ener
        cumen2 += ener**2
        
    # Compute mean energy and std 
        
    mean_energy = cumen / N_eq
    std_energy = m.sqrt( (1 / (N_eq-1) ) * (cumen2 / N_eq - mean_energy**2) )
        
    print("\n La frazione di proposte accettate è: {}".format(N_acc/N_eq))
    print("\n L'energia è: {} pm {}".format(mean_energy, std_energy))

    return part, mean_energy, std_energy


# EXECUTION OF THE PROCEDURE

# Set initial conditions

particles = initcond(particles)

# Let us verify our success with a plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2])
plt.title("Initial Conditions")

# Run the MC simulation

particles, mean_energy, std_energy = montecarlo(particles)

# Verify graphically that the simulation actually changed where the particles are
# Why? because if it does not do so then it surely does not work 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particles[:,0], particles[:,1], particles[:,2])
plt.title("New Configuration")
