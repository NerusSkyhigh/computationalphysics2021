import math as mt
import numpy as np
import time
from scipy.integrate import quad
import matplotlib.pyplot as plt

Z = 4 # For Berillium

# # Coefficients for 2G expansion

# alfa = np.array([[0.1153566907E+02, 0.2053343279E+01], 
#                   [0.5081630247E+00, 0.1288835617E+00]])

# Co = np.array([[0.4301284983E+00, 0.6789135305E+00],
#                 [0.4947176920E-01, 0.9637824081E+00]])

# # Coefficients for 3G expansion

# alfa = np.array([[0.3016787069E+02, 0.5495115306E+01, 0.1487192653E+01], 
#                   [0.1314833110E+01, 0.3055389383E+00,  0.9937074560E-01]])

# Co = np.array([[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
#                 [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]])

# Coefficients for 4G expansion

alfa = np.array([[0.7064859542E+02, 0.1292782254E+02, 0.3591490662E+01, 0.1191983464E+01],
                [0.3072833610E+01, 0.6652025433E+00, 0.2162825386E+00, 0.8306680972E-01]])

Co =  np.array([[0.5675242080E-01, 0.2601413550E+00, 0.5328461143E+00, 0.2916254405E+00],
                [-0.6220714565E-01 ,0.2976804596E-04, 0.5588549221E+00, 0.4977673218E+00]])

# Coefficients for 5G expansion

# alfa = np.array([[0.1531054580E+03, 0.2805617168E+02, 0.7836289220E+01, 0.2675605246E+01, 0.1008268481E+01], 
#                   [0.6660499143E+01, 0.1365485848E+01, 0.4351816281E+00, 0.1691742165E+00, 0.7206945827E-01] ])

# Co = np.array([[0.2214055312E-01, 0.1135411520E+00, 0.3318161484E+00, 0.4825700713E+00, 0.1935721966E+00],
#                 [-0.2940855088E-01, -0.6532746883E-01, 0.1289973181E+00, 0.6122899938E+00, 0.3461205655E+00]])

# # Coefficients for 6G expansion

# alfa = np.array([[0.3128704937E+03, 0.5736446253E+02, 0.1604850940E+02, 0.5513096119E+01, 0.2140896553E+01, 0.8817394283E+00 ], 
#                   [0.1363324744E+02, 0.2698375464E+01, 0.8386530829E+00, 0.3226600698E+00, 0.1401314882E+00, 0.6423251387E-01] ])

# Co = np.array([[0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00, 0.3705627997E+00, 0.4164915298E+00, 0.1303340841E+00],
#                 [-0.1325278809E-01, -0.4699171014E-01, -0.3378537151E-01, 0.2502417861E+00, 0.5951172526E+00, 0.2407061763E+00]])

a = 0.5 # Mixing parameter
ncontr = 4 # Number of functions in the contracted orbital
delta = 1E-10 # Convergence threshold
dim = 2 # number of basis functions
Nelec = 4 # Number of electrons

# Computes the single particle, direct and exchange integrals with given orbitals

def integrals(alfa):

    # SINGLE PARTICLE HAMILTONIAN  (one electron only)

    H_mat = np.zeros((dim,dim))
    
    for i in range(0,dim):
       for j in range(0,dim):
           for k in range(0,ncontr):
               for l in range(0,ncontr):
                   
                  H_mat[i,j] += Co[i,k] * Co[j,l] * ( 3 * (alfa[i,k] * alfa[j,l] \
                            * mt.pi**(3.0/2.0) ) / (( alfa[i,k] +  alfa[j,l])**(5.0/2.0) ) \
                                                     - Z*2*mt.pi/(alfa[i,k]+alfa[j,l])) 

    # OVERLAP MATRIX

    S = np.zeros((dim,dim))

    for i in range(0,dim):
        for j in range(0,dim):
            for k in range(0,ncontr):
                for l in range(0,ncontr):
                    
                   S[i,j] += Co[i,k]*Co[j,l]*(mt.pi / (alfa[i,k] \
                                                   + alfa[j,l] ) )**(3.0/2.0)

    # 2-BODY POTENTIAL

    Exch_mat = np.zeros((int(dim**2), int(dim**2)))

    for i in range(0, dim):
        for j in range(0,dim):
            for k in range(0, dim):
                for l in range(0, dim):
                    for n in range(0, dim):
                        for o in range(0, dim):
                            for p in range(0, dim):
                                for q in range(0, dim):
                    
                                    Exch_mat[dim*i+j, dim*k+l] += Co[i,n]*Co[j,o]*Co[k,p]*Co[l,q]*\
                                    (2 * mt.pi**(5.0/2.0)) / ((alfa[i,n] + alfa[j,o]) * (alfa[k,p] + alfa[l,q]) \
                                          * mt.sqrt(alfa[i,n] + alfa[j,o] + alfa[k,p] + alfa[l,q] )) 

    return H_mat, S, Exch_mat


# Diagonalisation procedure

def diagonalisation(F, V):

    Fpr = np.dot(np.dot(np.transpose(V), F), V)

    eigvalF, eigvecF = np.linalg.eigh(Fpr)

    Cnew = np.dot(V, eigvecF)

    return eigvalF, Cnew

# Define function to compute the density matrix

def densitymatrix(P, C):

    P_old = np.zeros((dim, dim))

    for r in range(0, dim):

        for s in range(0, dim):

            P_old[r,s] = P[r, s]
            P[r,s] = 0
            
            for k in range(0, int(Nelec/2)):

                P[r,s] += C[r,k] * C[s,k]

    return P, P_old

# Define function that computes the fock operator

def fock(H, P, Exch): # Make Fock Matrix

    F = np.zeros((dim, dim))

    for i in range(0, dim):

        for j in range(0, dim):

            F[i,j] = H[i,j]

            for k in range(0, dim):

                for l in range(0, dim):

                    F[i,j] += P[k,l] *(2.0* Exch[dim*j + i, dim*l + k] \
                                               - Exch[dim*l +i, dim*j + k] )

    return F

# Function to calculate the energy

# def energy(P, H, F, En):

#     EN = 0

#     for r in range(0, dim):
#         for s in range(0, dim):

#             EN += P[r,s] * ( H[r,s] + F[r,s] )

#     return EN


def energy(P, H, F, En):

    EN = 0 
    
    for i in range(0,2):
        
        EN += En[i]
        
    EN += (P*H).sum()

    return EN



# Calculate change in density matrix using Root Mean Square Deviation (RMSD)

def deltap(D, Dold):

    DELTA = 0.0

    for i in range(0, dim):

        for j in range(0, dim):

            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA)**(0.5)

# Function for iterative process

def iteration(H, Exch, V):

    i = 0

    Enewt = -90
    diff = 1

    P_old = np.zeros((dim,dim))
    P = P_old

    while (diff > delta):

        P_mix = a * P + (1 - a) * P_old

        F = fock(H, P_mix, Exch)

        Enew, Cnew = diagonalisation(F, V)

        # Calculate the density matrix

        P, P_old = densitymatrix(P, Cnew)

        Enewt = energy(P, H, F, Enew)

        diff = deltap(P, P_old)

        print("Step: " + str(i) + ", Energy: " + str(Enewt) )

        i += 1

    return Enewt, Cnew, F, P, Enew

# Start timing

start = time.time()

# Calculate integrals

H, S ,Exch = integrals(alfa)

# Diagonalise S matrix

eigvalS, eigvecS = np.linalg.eigh(S)

# Construct diagonal eigenvalues matrix

Seighlf = ( np.diag(eigvalS**(-0.5) ) )

# V MATRIX CONSTRUCTION

V = np.dot(eigvecS, Seighlf)

# RUN CALCULATION

Enewt, Cnew, F, P, Enew = iteration(H, Exch, V)

# End timing

end = time.time()

# Print results

print("\nRESULTS USING STO-{}G CONTRACTED ORBITALS".format(ncontr))

print("\nThe orbital energies are: ")

print("\n{:.6f} Hartrees for the 1s \n{:.6f} Hartrees for the 2s".format(Enew[0], Enew[1]))

print("\nThe GS energy is {:.6f} hartrees".format(Enewt) )

print("\nThe calculation took {:.5f} seconds".format(end-start))

# Define the integrand function for normalisation

def G(x, a, b):
    return a * mt.exp(-b * x**2)

def GT2(x, C1, C2, Co1, alfa1, Co2, alfa2):
    Gtot2 = 0
    for i in range(0, ncontr):
        
        Gtot2 += C1 * G(x, Co1[i], alfa1[i]) + C2 * G(x, Co2[i], alfa2[i])
        
    return Gtot2**2

# Verify this is normalised

Co1s = Co[0,:]
Co2s = Co[1,:]
alfa1s = alfa[0,:]
alfa2s = alfa[1,:]

I1s = quad(GT2, 0, np.inf, args=(Cnew[0,0], Cnew[1,0], Co1s, alfa1s, Co2s, alfa2s))
I2s = quad(GT2, 0, np.inf, args=(Cnew[0,1], Cnew[1,1], Co1s, alfa1s, Co2s, alfa2s))

print("\nThe norm for the 1s wf is {:.6f} pm {} ".format(I1s[0],I1s[1]))
print("\nThe norm for the 2s wf is {:.6f} pm {} ".format(I2s[0],I2s[1]))

# Functions Drawing

def Gdr(x, a, b):
    return a * np.exp(-b * np.power(x,2)) 

def GT(x, C, Co, alfa):
    Gtot2 = 0
    for i in range(0, ncontr):
        Gtot2 += Gdr(x, Co[i], alfa[i])
        
    return  Gtot2 * C

def GT2(x, C1, C2, Co1, alfa1, Co2, alfa2):
    Gtot2 = 0
    for i in range(0, ncontr):
        
        Gtot2 += C1 * Gdr(x, Co1[i], alfa1[i]) + C2 * Gdr(x, Co2[i], alfa2[i])
        
    return Gtot2**2 

r = np.arange(0,5,0.01)

# Plot the gaussian, the total wavefunction and the exact result

orb1s = (GT(r, Cnew[0,0], Co1s, alfa1s) + GT(r, Cnew[1,0], Co2s, alfa2s)) / mt.sqrt(I1s[0])

orb2s = (GT(r, Cnew[0,1], Co1s, alfa1s) + GT(r, Cnew[1,1], Co2s, alfa2s)) / mt.sqrt(I2s[0])

fig = plt.figure()
plt.plot(r, orb1s, label = "GTO Orbital, 1s")
plt.plot(r, orb2s, label = "GTO Orbital, 2s")
plt.legend()
plt.grid()
plt.title("Orbitals with Contranctions")
plt.savefig("Gaussians_Berillium_contr.png", dpi=300)

# Electronic density plot

orb1s2 = GT2(r, Cnew[0,0], Cnew[1,0], Co1s, alfa1s, Co2s, alfa2s) / I1s[0]

orb2s2 = GT2(r, Cnew[0,1], Cnew[1,1], Co1s, alfa1s, Co2s, alfa2s) / I2s[0]

fig = plt.figure()
plt.plot(r, orb1s2, label="Density 1s")
plt.plot(r, orb2s2, label="Density 2s")
plt.legend()
plt.grid()
plt.title("Berillium Electronic Density (contractions)")
plt.savefig("Density_Be_contr.png", dpi=300)
