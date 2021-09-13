# DFT
import time
import numpy as np
from numba import njit
import math as m
import matplotlib.pyplot as plt


# ------------------ Define system parameters  ------------------
Ne = 18 # [], Number Ef electrons. Try with 2, 8, 18, 20

# Radius of the Jellium sphere normalized to the atom
rs = 3.93 # [Bhor radius], for Na
#rs = 4.86 # [Bhor radius], for K

# RHO Blob: density of the Jellium
rhob = 3.0 / (4.0 * m. pi * rs**3)  # [1/Bhor radius^3]

# Actual radius of the Jellium sphere due to multiple electrons
# Rc = (3.0 * Ne / (4.0 * m. pi * rhob) )**(1.0 / 3.0)
Rc = Ne**(1/3) * rs # [1/Bhor radius^3]


# --------------- Parameters of the computation  ---------------
Nx = 500 # Number of mesh points
dE = 0.00001 # energy step [?]
L = 14.0 # Mesh length [Bhor radius]
dx = L / Nx # mesh spacing [Bhor radius]

# La documentazione mi dice che np.arange è inconsistente se si usano range non
# interi. https://numpy.org/doc/stable/reference/generated/numpy.arange.html
# Uso numpy.linspace come consigliato. Non dovrebbe cambiare nulla ma potrebbe
# evitarci dei problemi
# - Guglielmo
# x = np.arange(0, dx * Nx, dx)
x = np.linspace(start=0, stop=L, num=Nx, endpoint=False)


# Correlation function parameters ?
p = 1.0
A = 0.031091
a1 = 0.21370
b1 = 7.5957
b2 = 3.5876
b3 = 1.6382
b4 = 0.49294

# Mixing parameter

a = 0.01

# Define a function that computes the direct term in the potential
@njit
def direct(r, rho):
    u = 0.0

    for i in range(0, Nx):
        rpr = dx * i

        if rpr <= r and r != 0:

            u += (1 / r) * rho[i] * rpr**2 * dx

        else:
            # Outside of the sphere we see only the total Density
            u += rho[i] * rpr * dx


    return 4 * m.pi * u

# Function for the exchange term
@njit
def exchange(r, rho):
    ind = int(r/dx)

    # Pad density with zeros to avoid naugthy divergences
    ex = -(3.0 * rho[ind] / m.pi)**(1.0 / 3.0)

    return ex

# Function for the correlation term

@njit
def correlation(r):
    G = -2.0 * A * (1 + a1 * rs) * m.log(1.0 + 1.0 / (2.0 * A * \
        (b1 * rs**(1.0/2.0) + b2 * rs + b3 * rs**(3.0/2.0) + b4 * r**(p + 1.0) ) ) )
    return G


# Define a function for the potential. Takes as input, position r, l quantum # and mixed density
@njit
def potential(r, l, rho):
    V = direct(r, rho) + exchange(r, rho) + correlation(r)

    if r <= Rc:
        V += 2.0 * m.pi * rhob * ( (1.0 / 3.0) * r**2 - Rc**2 )

    else:
        V += - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / r)

    if r != 0:
        V += l * (l + 1) / (2 * r**2)

    return V


# Define a potential function for the drawing phase
# Takes as input position r and mixed density rho
@njit
def draw_pot(r, rho):
    V = direct(r, rho) + exchange(r, rho) + correlation(r)

    if r <= Rc:
        V += 2.0 * m.pi * rhob * ( (1.0 / 3.0) * r**2 - Rc**2 )

    else:
        V += - 2.0 * m.pi * rhob * (2.0 / 3.0) * (Rc**3 / r)

    return V


# Define Numerov function
@njit
def numerov(psi1, psi2, i, E, l, rho):

    p1 = potential(dx * (i-2), l, rho)
    p2 = potential(dx * (i-1), l, rho)
    pf = potential(dx * i, l, rho)

    k1 = 2.0 * (E - p1)
    k2 = 2.0 * (E - p2)
    kf = 2.0 * (E - pf)

    num1 =  psi2 * (2.0 - (5.0 / 6.0) * k2 * dx**2)
    num2 = psi1 * (1.0 + (1.0 / 12.0) * k1 * dx**2)
    den = 1.0 + (1.0 / 12.0) * kf * dx**2

    psif = (num1 - num2) / den

    return psif


# Function to construct the density
@njit
def rhoconstr(psi, Nel):

    rho = np.zeros(Nx)

    for i in range(1,Nx,1):

        rho[i] = (psi[i] / x[i])**2

    rho[0] = rho[1]

    return rho

# Function to integrate for normalisation

@njit
def normalisation(x, psi, rho):

    I = 0

    for i in range(0,Nx):

        I += rho[i] * dx

    No = I * rs / (rhob * Rc)

    rho = rho / No

    psi = psi / No**0.5

    return psi, rho


@njit
def level(n, l, En, last, rhoin):

    psi = np.zeros(Nx)

    # Number of electrons

    if l == 0:
        Nel = 2
    elif l == 1:
        Nel = 6
    elif l == 2:
        Nel = 10

    psi[0] = 0
    psi[1] = pow(dx, l+1)

    if l == 2:

        psi[0] = 0
        psi[1] = dx


    rat = 1.0
    lastnow = last

    Ener = En

    while rat > 0:

            for i in range(2, Nx, 1):

                psi[i] = numerov(psi[i-2], psi[i-1], i, Ener, l, rhoin)


            lastbefore = lastnow
            lastnow = psi[Nx-1]
            rat = lastnow / lastbefore

            Ener = Ener + dE


    zero = Ener- dE - (dE / (lastnow - lastbefore) ) * lastnow

    # Construct new density

    rho = rhoconstr(psi, Nel)

    psi, rho = normalisation(x, psi, rho)

    return zero, psi, lastnow, rhoin, rho

# SELF CONSISTENT PROCEDURE

@njit
def selfcons(n, l, En, last):

    tr = 1E-4

    # Set initial densities

    rhold = np.zeros(Nx)
    rhonew =  np.zeros(Nx)

    Eold = 0
    energy = En

    while abs(Eold - energy) > tr:

        Eold = energy

        rhomix = a*rhonew + (1-a) * rhold

        energy, psi, lastnow, rhold, rhonew = level(n, l, En, last, rhomix)

    return energy, psi, lastnow, rhold, rhonew


# RUN SIMULATION

begin = time.time()

# We set an initial rho for testing

rho = np.zeros(Nx)

# Construct 1s level

#En1s, psi1s, last1s, rho1so, rho1s = level(1, 0, - 2.0 * m.pi * rhob * Rc**2, 1.0, rho)

En1s, psi1s, last1s, rho1so, rho1s = selfcons(1, 0, - 2.0 * m.pi * rhob * Rc**2, 1.0)

print("\nThe energy of the 1s is {}".format(En1s))

 # Construct 1p level

En1p, psi1p, last1p, rho1po, rho1p = selfcons(1, 1, En1s, -last1s)

print("\nThe energy of the 1p is {}".format(En1p))

 # Construct 1p level

En1d, psi1d, last1d, rho1do, rho1d = selfcons(1, 2, En1p, -last1p)

print("\nThe energy of the 1d is {}".format(En1d))

# # # Construct 2s level

# # En2s, psi2s, rho2s, last2s = level(2, 0, En1d, -last1d)

# # print("\nThe energy of the 2s is {}".format(En2s))

# Construct final density

rho =  rho1s + rho1p + rho1d #+ rho2s

end = time.time()

print("\nThis took me "+str(end-begin)+" seconds.")




# ------------------ Plot Results  ------------------
from datetime import datetime

# WF figure
fig = plt.figure()

plt.plot(x[0:int(14/dx)], psi1s[0:int(14/dx)], linewidth=0.9, label="1s")
plt.plot(x[0:int(14/dx)], psi1p[0:int(14/dx)], linewidth=0.9, label ="1p")
plt.plot(x[0:int(14/dx)], psi1d[0:int(14/dx)], linewidth=0.9, label="1d")
# plt.plot(x[0:int(14/dx)], psi2s[0:int(14/dx)], linewidth=0.9, label="2s")

plt.legend()
plt.title("Wavefunction with {} electrons".format(Ne))
plt.axvline(Rc, color = "grey")
plt.savefig("WF_20_el_"+str(datetime.now())+".png", dpi=300)


# Density figure
fig1 = plt.figure()
plt.plot(x[0:int(14/dx)], rho[0:int(14/dx)], linewidth=0.9)
plt.title("Density with {} electrons".format(Ne))
plt.axhline(rhob, xmin = 0, xmax = Rc/14.24, color="grey", linestyle="-.", linewidth=0.9)
plt.axhline(0, xmin = Rc/14.24, xmax = 1, color="grey", linestyle="-.", linewidth=0.9)
plt.axvline(Rc, ymin = 0.052, ymax = 0.6, color="grey", linestyle="-.", linewidth=0.9)
plt.savefig("rho_20_el_"+str(datetime.now())+".png", dpi=300)

# Potential
@njit
def drawvec():

    V0 = np.zeros(Nx)
    di = np.zeros(Nx)
    co = np.zeros(Nx)
    ex = np.zeros(Nx)

    for i in range(0, Nx):
        V0[i] = draw_pot(x[i], rho)
        di[i] = direct(x[i], rho)
        co[i] = correlation(x[i])
        ex[i] = exchange(x[i], rho)

    return V0, di, co, ex

V0, di, co, ex = drawvec()

fig2 = plt.figure()
plt.plot(x, V0, linewidth=0.9, label="Total")
plt.plot(x, di, linewidth=0.9, color="black", label="direct")
plt.plot(x, co, linewidth=0.9, color="magenta", label="correlation")
plt.plot(x, ex, linewidth=0.9, color="green", label="exchange")
plt.title("Potential")
# plt.axhline(En1s , color="green", linestyle="-.", label="1s", linewidth=0.9)
# plt.axhline(En1p , color="blue", linestyle="-.", label="1p", linewidth=0.9)
# plt.axhline(En1d , color="red", linestyle="-.", label="1d", linewidth=0.9)
# plt.axhline(En2s , color="cyan", linestyle="-.", label="2s", linewidth=0.9)
plt.legend()
plt.savefig("Pot_20_el_"+str(datetime.now())+".png", dpi=300)
