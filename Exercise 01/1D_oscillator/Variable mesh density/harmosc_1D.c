#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

 // Global variables

double deltaE = 0.00001; //Energy increment, set to 10^-5 for final simulation.
double Emin = 0.; // Potential minumum
double xmin = 0.; // x minumum
double L = 6.; // lunghezza mesh
int Nx = 500; // Number of points in mesh
double deltax; // Mesh spacing

// POTENTIAL FUNCTION

double potential(double x)
  {
    double p;

    p = (1. / 2.) * x * x;

    return p;
  }

//NUMEROV FUNCTION

double numerov(double psi1, double psi2, int ind, double En)
  {
    double f;
    double k1, k2, kf, p1, p2, pf; //already squared
    double valn1, valn2, vald;

    p1 = potential(deltax * (ind - 2));
    p2 = potential(deltax * (ind - 1));
    pf = potential(deltax * ind);

    k1 = 2. * (En - p1);
    k2 = 2. * (En - p2);
    kf = 2. * (En - pf);

    valn1 = psi2 * (2 - (5. / 6.) * deltax * deltax * k2);
    valn2 = psi1 * (1 + (1. / 12.) * deltax * deltax * k1);
    vald = 1 + (1. / 12.) * deltax * deltax * kf;

    f = (valn1 - valn2) / vald;

    return f;
  }


// MAIN FUNCTION

int main()
{
    double psi[Nx]; // Vector for psi
    int i, n, j;
    double rat, Eexact, restart=0;
    double Ener = Emin;
    double valatt, valprec;
    double zero;
    int nmax = 5;

    deltax = L / Nx;

    for(n=0; n< nmax; n++){

      j = 0;
      rat = 1;

      // Initial conditions

      if(n==0){
      psi[0] = 1.;
      psi[1] = 1.;
      }else if(n==1){
        psi[0] = 0.;
        psi[1] = -1. * deltax;
      } else if(n==2){
        psi[0] = -1.;
        psi[1] = -1.;
      } else if(n==3){
        psi[0] = 0.;
        psi[1] = deltax;
      } else if(n==4) {
        psi[0] = 1.;
        psi[1] = 1.;
      }

      do{

        for(i = 2; i < Nx; i++) {

            psi[i] = numerov(psi[i-2], psi[i-1], i, Ener);
            //vecchio = nuovo;
            //nuovo = psi[i];
            //printf("%lf, %lf, %lf\n", psi[i],p3, (nuovo-vecchio)*10000);
        }

        valprec = valatt;
        valatt = psi[Nx-1];

        rat = valatt / valprec;

        //printf("%lf,   %lf\n", prod, Ener);

        Ener += deltaE;

      } while(rat > 0);

      //printf("%lf, %lf\n", Ener-deltaE, Ener);
      //printf("%lf, %lf\n", valprec, valatt);

      // Refinement

      zero = (Ener-deltaE) - ( deltaE / (valatt - valprec) ) * valatt;

      // Exact energy of the 3D harmonic oscillator

      Eexact =  n  + (1. / 2.);

      printf("%.10lf, %.10lf\n", zero, Eexact);

    }

    return 0;
}
