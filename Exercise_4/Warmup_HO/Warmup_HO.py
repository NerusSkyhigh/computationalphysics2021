# MONTE CARLO TEST WITH HARMONIC OSCILLATOR iN 1D

import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt
import random as rd
 
N_eq = 5000 # Equilibration steps

N_steps = 50000 # Simulation steps

alpha = 3.0 # Variational parameter for the WF

# Function to compute the WF

@njit 
def WF(x, alpha):
    
    # We can forget about the normalisation, it comes "for free" in MC
    
    Psi = m.exp(- x**2 / (2*alpha**2) ) #* 1 / (2*m.pi*alpha**2)**0.5
    
    return Psi


# FUNCTION FOR THE METROPOLIS ALGORITHM

@njit
def metropolis(x, N_acc, wf2_old, Delta):

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
def k_energy(x):
    
    k_en = - 0.5 * (-1 / alpha**2 + x**2 / alpha**4) 
    
    return k_en

# FUNCTION TO PERFORM THE ITERATIVE PROCEDURE

#@njit
def montecarlo(x):
    
    Delta = 4 # Value of parameter for displacement
    
    # vectors to do the plot in the end 
    
    pot = np.zeros(N_eq+N_steps)
    kin = np.zeros(N_eq+N_steps)
    tot = np.zeros(N_eq+N_steps)

    # First, the equilibration steps

    wf2 = WF(x,alpha)

    N_acc1 = 0
    
    # Equilibration with fixed steps
    
    for s in range(0,N_eq): # stop the equilibration when variance is smaller than a value
    
        x, N_acc1, wf2 = metropolis(x, N_acc1, wf2, Delta)
        
        # Vectors for plotting
        pot[s] = p_energy(x)
        kin[s] = k_energy(x)
        tot[s] = pot[s] + kin[s]
    
        # An adaptive scheme for Delta 
        
        if s != 0  and s%5 == 0:
        
            
            if N_acc1 / s >= 0.55:
                
                Delta += 0.05 * 4
                
            elif N_acc1 / s <= 0.45:
                
                Delta -= 0.05 * 4
        

    # # Equilibration with standard deviation

    # while std / mean > 0.02: # stop the equilibration when variance is smaller than a value
    
    #     x, N_acc1, wf2 = metropolis(x, N_acc1, wf2, Delta)
        
    #     # Vectors for plotting
    #     pot[s] = p_energy(x)
    #     kin[s] = k_energy(x)
    #     tot[s] = pot[s] + kin[s]
        
    #     # Instead of using a chosen number of equilibration steps, we should
    #     # use the convergence of the energy to stop the procedure 
        
    #     # Cumulate the observables
    #     loc_en = k_energy(x) + p_energy(x)
    #     cum_en += loc_en
    #     cum_en2 += loc_en**2
        
    #     if s >= 2 and s%5 == 0:
            
    #         mean = cum_en / s
    #         std = m.sqrt( 1 / (s-1) * ( cum_en2 / s - mean**2) )
            
    #     s += 1
            
        
    # Then, the simulation steps

    N_acc2 = 0
    cum_en = 0 # Reset cumulative variable
    cum_en2 = 0 # Reset cumulative variable

    for s in range(0, N_steps):

        x, N_acc2, wf2 = metropolis(x, N_acc2, wf2, Delta)
        
        # Vectors for plotting
        pot[s+N_eq] = p_energy(x)
        kin[s+N_eq] = k_energy(x)
        tot[s+N_eq] = pot[s+N_eq] + kin[s+N_eq]
        
        # Cumulate the observables
        loc_en = k_energy(x) + p_energy(x)
        cum_en += loc_en
        cum_en2 += loc_en**2
        
         # An adaptive scheme for Delta 
        
        if s != 0  and s%10 == 0:
        
            if N_acc2 / s >= 0.55:
                
                Delta += 0.05 * 4
                
            elif N_acc2 / s <= 0.45:
                
                Delta -= 0.05 * 4

    accf = N_acc2/N_steps
    
    mean_en = cum_en / N_steps
    std_en = m.sqrt( 1 / (N_steps-1) * ( cum_en2 / N_steps - mean_en**2) )

    return x, accf, pot, kin, tot, mean_en, std_en


# EXECUTION OF THE PROCEDURE

# Start procedure timing

start = time.time()

# Set initial conditions

x = 0

# Run the MC simulation

x, accf, pot, kin, tot, mean, std = montecarlo(x)

print("\n The fraction of accepted moves was {}".format(accf))
print("\n The energy is {} pm {}".format(mean, std))

# Plot energies at each step

# NB we notice that here zeros appear in the end of the equilibration region.
# This is due to the fact that to use njit we needed to set the length of the 
# array before filling it. This is inconvenient here because we do not know how 
# many steps the equilibration phase will take... if they are fewer than the 
# equilibration phase steps, zeroes will remain in the plot. If they exceed the
# number, we simply will not see them. This is just a problem of doing the plot
# that i simply am too lazy to solve now.

steps = np.arange(0,N_eq+N_steps)

fig = plt.figure()
plt.plot(steps, pot, label = "Potential")
plt.plot(steps, kin, label = "Kinetic")
plt.plot(steps, tot, label = "Total")
plt.xlabel(r"$N_{steps}$")
plt.ylabel("Energy")
plt.title("Energies")
plt.legend()

# Only total energy

fig = plt.figure()
plt.plot(steps, tot)
plt.xlabel("$N_{steps}$")
plt.ylabel("Energy")
plt.title("Total Energy")

# End procedure timing

end = time.time()

print("\n The simulation took me {} seconds".format(end-start))