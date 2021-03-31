#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<float.h>
#include "bessel.h"

 // Global variables

double deltaE = 0.00001; //Energy increment, set to 10^-5 for final simulation.
double L = 20.; // lunghezza mesh
int Nx = 2000; // Number of points in mesh
double deltax; // Mesh spacing
double fac = 0.03528; // hbar^2 / 2m in units of sigma and epsilon

// POTENTIAL FUNCTION
// CONTAINS LENNARD JONES INTERATOMIC TERM AND ANGULAR MOMENTUM

double potential(double r, int a)
  {
    double p;
    double t1, t2, lenn;

    if(r == 0 ){

      r = deltax;

    }else{

      t1 = 1. / pow(r, 12);
      t2 = 1. / pow(r, 6);
      lenn = t1 - t2;
      p = 4 * lenn + fac * (a * (a + 1) ) / (r * r);

    }

    return p;
  }

//NUMEROV FUNCTION

double numerov(double psi1, double psi2, int ind, double En, int ang)
  {
    double f;
    double k1, k2, kf, p1, p2, pf; //already squared
    double valn1, valn2, vald;

    p1 = potential(deltax * (ind - 2), ang);
    p2 = potential(deltax * (ind - 1), ang);
    pf = potential(deltax * ind, ang);

    k1 = (1. / fac) * (En - p1);
    k2 = (1. / fac) * (En - p2);
    kf = (1. / fac) * (En - pf);

    valn1 = psi2 * (2 - (5. / 6.) * deltax * deltax * k2);
    valn2 = psi1 * (1 + (1. / 12.) * deltax * deltax * k1);
    vald = 1 + (1. / 12.) * deltax * deltax * kf;

    f = (valn1 - valn2) / vald;

    return f;
  }

// PHASE SHIFT function

double phaseshift(int n1, int n2, int l, double psi1, double psi2, double Energy)
{
    double Kbig, Ksmall;
    double num1, num2, den1, den2, phsh;

    Kbig = ( psi1 * (deltax * n2) ) / ( psi2 * (deltax * n1) );
    Ksmall = sqrt(Energy / fac);

    num1 = Kbig * besselj(l, Ksmall * (deltax * n2));
    num2 = besselj(l, Ksmall * (deltax * n1) );
    den1 = Kbig * besseln(l, Ksmall * (deltax * n2));
    den2 = besseln(l, Ksmall * (deltax * n1) );
    phsh = atan( (num1 - num2) / (den1 - den2) );

    return phsh;

}

// MAIN FUNCTION

int main()
{
    double psi[Nx]; // Vector for psi
    int i, l, j, N1, N2;
    double shift, k, En;
    double valatt= 1., valprec=0.;
    double zero, point1,point2, b, crs;
    int lmax = 6;
    char filename[20];
    FILE *fp;

    deltax = L / Nx;

    fp = fopen("CrossSection.csv", "w+"); //Open file

    for (En = deltaE; En < 0.59 ; En+= deltaE) {

      crs = 0.;

      for(l=0; l <= lmax; l++){

        //printf("Phase Shift l=%i\n", l); //Print on screen l for visual check

        b = 1.1632127145;
        psi[0] = exp(-1.0 * pow( b / 0.5, 5) );
        psi[1] = exp(-1.0 * pow( b / (0.5+deltax), 5) );

        for(i = 2; i < Nx; i++) {

            psi[i] = numerov(psi[i-2], psi[i-1], i + (int)(0.5/deltax), En, l);

        }

          N1 = 1710;
          N2 = 1720;

          shift = phaseshift(N1, N2, l, psi[N1], psi[N2], En);

          //printf("%d \t %d \t %lf\n",N1,N2, shift);

          crs += pow(sin(shift),2) * (2*l+1) * ( (4 * M_PI) / (En / fac) );

      }

      printf("%lf\n",crs);
      fprintf(fp, "%lf\n",crs);

    }

      fclose(fp);

    return 0;
}
