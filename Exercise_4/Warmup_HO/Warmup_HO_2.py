# MONTE CARLO TEST WITH HARMONIC OSCILLATOR iN 1D


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

# FUNCTION TO PERFORM THE ITERATIVE PROCEDURE

@njit
def montecarlo(x, alpha):
    
    Delta = 4 # Value of parameter for displacement

    # First, the equilibration steps

    wf2 = WF(x,alpha)

    N_acc1 = 0
    cum_en = 0
    cum_en2 = 0 
    std = 1 
    mean = 1
    s = 0
    
    while std / mean > 0.02: # stop the equilibration when variance is smaller than 1%
        
        # An adaptive scheme for Delta would be needed

        x, N_acc1, wf2 = metropolis(x, N_acc1, wf2, Delta, alpha)
        
        # Instead of using a chosen number of equilibration steps, we have to
        # use the convergence of the energy to stop the procedure 
        
        # Cumulate the observables
        loc_en = k_energy(x, alpha) + p_energy(x)
        cum_en += loc_en
        cum_en2 += loc_en**2
        
        if s >= 2 and s%5 == 0:
            
            mean = cum_en / s
            std = m.sqrt( 1 / (s-1) * ( cum_en2 / s - mean**2) )
            
        s += 1
        
    # If we do things in the following way, we choose the length of the 
    # simulation. Alternatively we can choose again the target accuracy
    # The problem in doing so is that for some values of the variational 
    # parameter (too far from exact) a high accuracy may take forever to reach

    # Then, the simulation steps

    N_acc2 = 0
    cum_en = 0 # Reset cumulative variable
    cum_en2 = 0 # Reset cumulative variable


    for s in range(0, N_steps):

        x, N_acc2, wf2 = metropolis(x, N_acc2, wf2, Delta, alpha)
        
        loc_en = k_energy(x, alpha) + p_energy(x)
        cum_en += loc_en
        cum_en2 += loc_en**2
        
    mean_en = cum_en / N_steps
    std_en = m.sqrt( 1 / (N_steps-1) * ( cum_en2 / N_steps - mean_en**2) )

    accf = N_acc2/N_steps
    
    return accf, mean_en, std_en

# FUNCTION TO IMPLEMENT THE VARIATIONAL PROCEDURE

# Option 1: Re-run the MC simulation using a different variational parameter 
# each time. This has the advantage of working in any range of parameters but 
# requires us to run the simulation multiple times

@njit
def variational_1(x):
    
    alpha = np.arange(0.15, 3, 0.05) # vector for parameters to try
    
    l = len(alpha)
    
    accept = np.zeros(l)
    mean = np.zeros(l)
    std = np.zeros(l)    
    
    # run the simulation for all the parameters 
    
    for i in range(0, l):
        
        accept[i], mean[i], std[i] = montecarlo(x, alpha[i])
        
    return alpha, accept, mean, std

# Option 2: Run the simulation only once and use reweighting to compute the 
# result for different variational parameters. It has the advantage of running
# a single simulation but only works in a range of parameters ( of course since 
# we usually have an idea of rhe result, this is the preferred choice)

@njit
def variational_2(x):
    
    alpha_0 = 1 
    
    alpha_v = np.arange(0.8, 1.2, 0.05) # parameters to try
    
    return alpha, accept, mean, std


# EXECUTION OF THE PROCEDURE

# Start procedure timing

start = time.time()

# Set initial conditions

x = 0

# Run the MC simulation

alpha, accept, mean, std = variational_1(x)

print("\n The energy is {} pm {}".format(np.min(mean), std[np.argmin(mean)]))

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

# Plot the acceptance probability as a function of alpha

fig, ax= plt.subplots(1, figsize=(8,5.5))
plt.plot(alpha, accept, linewidth = 0, marker=".", ms = 3, mec="blue", mfc="blue")
plt.grid()
# plt.xlim([0.9,1.1])
# plt.ylim([0.49, 0.51]) # Uncomment to see that tiny errorbars are actually there
plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel(r"$p_{acc}$", fontsize=14)
plt.title("Acceptance probability", fontsize=18)
plt.savefig("Acceptance Probability.png", dpi = 300)

# End procedure timing

end = time.time()

print("\n The simulation took me {} seconds".format(end-start))
