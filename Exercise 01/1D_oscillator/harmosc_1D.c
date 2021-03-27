#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

 // Global variables
double deltaE = 0.00001; //Energy increment, set to 10^-5 for final simulation.
double Emin = 0.; // Starting point for the energy.

double xmin = 0.; // Starting point of the mesh
double deltax = 0.001; //delta x = 10^-3 for the mash
int Nx = 6000; // Number of points in mesh

// POTENTIAL FUNCTION
double potential(double x)
{
  /*
    This function implements the simple Harmonic Potential
  */
  double p;
  p = (1. / 2.) * x * x;
  return p;
}


//NUMEROV FUNCTION
double numerov(double psi1, double psi2, int ind, double En)
{
  /*
    This function implmentes the code for the Numberov's method.
    https://en.wikipedia.org/wiki/Numerov%27s_method
    Parameters:
      double psi1: WF in the point i-2
      double psi2: WF in the point i-1
      int ind >=2: index of the point to evaluate
      double En: Possible energy of the eigenstate
    !:The index of the WF, k and p are evaluated in the mesh's
      order. On the other hand the values val... are in the
      formula's order.
  */
  double f; // Final result

  // k_i = (2m/hbar^2) * ( E-V(x_i) ) with m = hbar = 1
  // k_i are the squared of the usual value
  double k1, k2, kf, p1, p2, pf;

  // Prefactors of the terms in the Numberov's method
  // VALue of term "Number 1/2" or "Division"
  double valn1, valn2, vald;

  // We use the potential for the points i, i-1 and i-2...
  p1 = potential(deltax * (ind - 2));
  p2 = potential(deltax * (ind - 1));
  pf = potential(deltax * ind);

  // ... to calculate the value of the function k in that
  // point of the mesh. (we set m=1)
  k1 = 2. * (En - p1);
  k2 = 2. * (En - p2);
  kf = 2. * (En - pf);

  // We set hbar=1
  valn1 = psi2 * (2 - (5. / 6.) * deltax * deltax * k2);
  valn2 = psi1 * (1 + (1. / 12.) * deltax * deltax * k1);
  vald = 1 + (1. / 12.) * deltax * deltax * kf;

  f = (valn1 - valn2) / vald;
  return f;
}


// MAIN FUNCTION
int main()
{
  double psi[Nx]; // Vector for psi (Mesh)
  int i, n, j; // various indices
  double rat, Eexact;
  double Ener = Emin;

  // VALore ATTuale e PRECedente used to find the Energy
  // of the stationary state. It is important that valatt
  // starts positive as start with n=EVEN (to undertand
  // why read the appropriate part)
  double valatt=1, valprec;

  double zero;
  int nmax = 5;

  for(n=0; n< nmax; n++){
    j = 0;
    rat = 1;

    // Initial conditions are evaluated by taking an
    // approximate solution in the limit x-->0.
    // https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#One-dimensional_harmonic_oscillator
    // or https://en.wikipedia.org/wiki/Hermite_polynomials
    if(n==0){
      // cos(x-->0) \simeq 1
      psi[0] = 1.;
      psi[1] = 1.;
    }else if(n==1){
      // cos'(x-->0)=-sin(x-->0) \simeq x
      psi[0] = 0.;
      psi[1] = -1. * deltax;
    }else if(n==2){
      // -sin'(x-->0)=-cos(x-->0) \simeq -1
      psi[0] = -1.;
      psi[1] = -1.;
    }else if(n==3){
      // -cos'(x-->0)=sin(x-->0) \simeq x
      psi[0] = 0.;
      psi[1] = deltax;
    }else if(n==4){
      // sin'(x-->0)=cos(x-->0) \simeq 1
      psi[0] = 1.;
      psi[1] = 1.;
    }

    do{
      // Here we generate the mesh given the starting conditions
      for(i = 2; i < Nx; i++) {
          psi[i] = numerov(psi[i-2], psi[i-1], i, Ener);
      }

      // Let's consider a solution for a n=EVEN. These function diverge
      // to positive value for E < E_true and to negative value for
      // E > E_true. When we find a change in the sign we know we have
      // just surpassed E_true.
      // For an ODD solution it's the opposite: - for E < E_true and +
      // for E>E_true. This allows change in behaviour allows  us to
      // keep valatt without reinitializing it at each cycle.
      valprec = valatt;
      valatt = psi[Nx-1];
      rat = valatt / valprec;

      Ener += deltaE;
    }while(rat > 0);

    // Once we find the stationary state we use the Secant method to
    // obtain a better esteem of the energy
    // https://en.wikipedia.org/wiki/Secant_method
    // As before leaving the do-while cycle we add a last deltaE we
    // subtract it to obtain a better accuracy.
    zero = (Ener-deltaE) - ( deltaE / (valatt - valprec) ) * valatt;

    // Exact energy of the 1D harmonic oscillator
    Eexact =  n  + (1. / 2.);
    printf("%lf, %lf\n", zero, Eexact);
  }

  return 0;
}
