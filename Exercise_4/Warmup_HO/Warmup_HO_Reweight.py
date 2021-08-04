# MONTE CARLO TEST WITH HARMONIC OSCILLATOR IN 1D

# The prevuìious code allowd us to test the algorithm and verify the properties
# of the local kinetic energy with some plots. However, the variational parameter
# in the WF had to be changed manually. Now, we actually want to automate the 
# variational procedure to find the best value of alpha via a minimisation of the energy

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import random as rd

N_steps = 50000 # Simulation steps
N_eq = 5000

# Function to compute the WF

@njit 
def WF(x, alpha):
    
    # We can forget about the normalisation, it comes "for free" in MC
    
    Psi = m.exp(- x**2 / (2*alpha**2) ) #* 1 / (2*m.pi*alpha**2)**0.5
    
    return Psi

# FUNCTION FOR THE METROPOLIS ALGORITHM

@njit
def metropolis(x, N_acc, wf2_old, Delta, alpha):

    # First, we need to build a new configuration
    # We start from the initial one

    new_x = x 

    # Variation of position

    rand = rd.uniform(0,1) # Pick random number
    new_x = x - Delta * (rand - 0.5) # Set new coordinate

    # Now, we must evaluate the acceptance ratio using the ratio of the square moduli of the WFs

    wf2_new = WF(new_x, alpha)**2

    acc = wf2_new / wf2_old

    # Decide acceptance

    xi = rd.uniform(0,1)

    wf2 = wf2_old

    if xi < acc: # new_part becomes part, we accept the MC move

       x = new_x

       N_acc += 1 # Increase the number of accepted moves

       wf2 = wf2_new

    return x, N_acc, wf2

# FUNCTION TO COMPUTE THE POTENTIAL ENERGY 

@njit
def p_energy(x):
    
    p_en = 0.5 * x**2
    
    return p_en

# FUNCTION TO COMPUTE THE KINETIC ENERGY

@njit
def k_energy(x, alpha):
    
    k_en = - 0.5 * (-1 / alpha**2 + x**2 / alpha**4) 
    
    return k_en

# FUNCTION FOR CALCULATION OF ALL THE ENERGIES WITH REWEIGHTING 

@njit
def reweighting(alpha, alpha_s, x):
    
    l = len(alpha)
    
    weight = np.zeros(l)
    loc_en = np.zeros(l)
    
    for i in range(0, l):
        
        # compute all the weights for all the 
        
        weight[i] = m.exp( (-x**2 / 2) * ( (1.0/alpha[i]**2) - (1.0/alpha_s**2) ) )
        #weight[i] = m.exp( x**2 * (alpha[i]-alpha_s)/alpha_s**2)
        
        loc_en[i] = (p_energy(x) + k_energy(x, alpha[i])) * weight[i] # Compute energy
        
    return weight, loc_en

# FUNCTION TO PERFORM THE ITERATIVE PROCEDURE

#@njit
def montecarlo(x):
    
    alpha_s = 1.0 # Parameter for which we run the simulation
    
    dalpha = np.arange(-0.1, 0.1, 0.01) # Amounts by which we use the 
    
    # Construct array of variational parameters
    
    alpha = np.zeros(len(dalpha)) 
    
    for i in range(0, len(dalpha)):
        
        alpha[i] = alpha_s + dalpha[i]
    
    Delta = 4 # Value of parameter for displacement

    # First, the equilibration steps

    wf2 = WF(x,alpha_s)

    N_acc1 = 0
                
    # Equilibration with chosen steps             
                
    for s in range(0,N_eq): # stop the equilibration when variance is smaller than 1%
        
        # An adaptive scheme for Delta would be needed

        x, N_acc1, wf2 = metropolis(x, N_acc1, wf2, Delta, alpha_s)
    
        # An adaptive scheme for Delta 
          
        if s != 0  and s%5 == 0:
          
            if N_acc1 / s >= 0.55:
              
                Delta += 0.05 * 4
             
            elif N_acc1 / s <= 0.45:
             
                Delta -= 0.05 * 4
                   
    # Simulation Steps

    N_acc2 = 0
    cum_en = np.zeros(len(dalpha)) # Reset cumulative variable
    cum_en2 = np.zeros(len(dalpha)) # Reset cumulative variable
    cum_w = np.zeros(len(dalpha))
    mean_en = np.zeros(len(dalpha))
    std_en = np.zeros(len(dalpha))

    for s in range(0, N_steps):

        x, N_acc2, wf2 = metropolis(x, N_acc2, wf2, Delta, alpha_s)
        
         # An adaptive scheme for Delta 
          
        if s != 0  and s%5 == 0:
          
            if N_acc2 / s >= 0.55:
              
                Delta += 0.05 * 4
             
            elif N_acc2 / s <= 0.45:
             
                Delta -= 0.05 * 4
                
        # Compute local energy and weight
        
        weight, loc_en = reweighting(alpha, alpha_s, x)
        
        #print(weight)
        
        for i in range(0,len(dalpha)):
            
            cum_w[i] += weight[i]
            cum_en[i] += loc_en[i] 
            cum_en2[i] += (loc_en[i])**2
        
    for i in range(0,len(dalpha)):
        
        mean_en[i] = cum_en[i] / cum_w[i] 
        # Had to drop an absolute value in to make stuff work
        std_en[i] = m.sqrt( 1 / (N_steps-1) * abs( cum_en2[i] / cum_w[i] - mean_en[i]**2) )

    accf = N_acc2/N_steps
    
    return alpha, accf, mean_en, std_en


# EXECUTION OF THE PROCEDURE

# Start procedure timing

start = time.time()

# Set initial conditions3

x = 0

# Run the MC simulation

alpha, accept, mean, std = montecarlo(x)

print("\n The energy is {} pm {}".format(np.max(mean), std[np.argmax(mean)]))

# Plot the energy as a function of the variational parameter

fig, ax= plt.subplots(1, figsize=(8,5.5))
plt.errorbar(alpha, mean, std, elinewidth=1, linewidth = 0, marker=".", ms = 3, mec="blue", mfc="blue", label="Simulation Result")
plt.grid()
plt.axhline(0.5, linewidth = 0.8, c= "red", label="Exact Value")
# plt.xlim([0.9,1.1])
# plt.ylim([0.49, 0.51]) # Uncomment to see that tiny errorbars are actually there
plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.title("Energy", fontsize=18)
plt.legend()
plt.savefig("Energies.png", dpi = 300)

# Plot the std as a function of alpha

fig, ax= plt.subplots(1, figsize=(8,5.5))
plt.plot(alpha, std, linewidth = 0, marker=".", ms = 3, mec="blue", mfc="blue")
plt.grid()
# plt.xlim([0.9,1.1])
# plt.ylim([0.49, 0.51]) # Uncomment to see that tiny errorbars are actually there
plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel("Std", fontsize=14)
plt.title("Standard Deviation", fontsize=18)
plt.savefig("Standard Deviation.png", dpi = 300)

# End procedure timing

end = time.time()

print("\n The simulation took me {} seconds".format(end-start))