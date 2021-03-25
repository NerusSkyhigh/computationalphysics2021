#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

 // Comment

 // Global variables

double deltaE = 0.00001; //Energy increment, set to 10^-5 for final simulation.
double Emin = 0.; // Potential minumum
double xmin = 0.; // x minumum
double L = 8.; // lunghezza mesh
int Nx = 5; // Number of points in mesh
double deltax; // Mesh spacing
double hbar = 1., m = 1.;

// POTENTIAL FUNCTION

double potential(double r, int a)
  {
    double p;

    if(r==0){

      p = 0;

    }else{

      p = (1. / 2.) * m * r * r + (1. / (2. * m)) * (a * (a + 1)) / (r * r);

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

    k1 = (2. * m) * (En - p1);
    k2 = (2. * m) * (En - p2);
    kf = (2. * m) * (En - pf);

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
    int i, n, l, j;
    double rat, Eexact, restart;
    double Ener = Emin;
    double valatt=0., valprec=0.;
    double zero;
    int lmax = 2;
    char filename[20];
    FILE *fp;

    deltax = L / Nx;

    for(l=0; l <= lmax; l++){

      sprintf(filename, "Levels l=%i.csv", l); // make filename
      fp = fopen(filename, "w+"); //Open file

      printf("Levels l=%i\n", l); //Print on screen l for visual check

      // Initial conditions for l even or odd

      psi[0] = 0.;
      psi[1] = pow(deltax, l+1);

      // Reset energy value for every l cycle

      if(l == 0){
        Ener = 0.;
      } else{
        Ener = restart;
      }

      for(n=l; n<l+3; n++){

        j = 0;

        do{

          for(i = 2; i < Nx; i++) {

              psi[i] = numerov(psi[i-2], psi[i-1], i, Ener, l);
              //vecchio = nuovo;
              //nuovo = psi[i];
              //printf("%lf, %lf, %lf\n", psi[i],p3, (nuovo-vecchio)*10000);
          }

          valprec = valatt;
          valatt = psi[Nx-1];

          rat = valatt / valprec;

          //printf("%lf\n", rat);

          Ener += deltaE;

        } while(rat > 0);

        //printf("%lf, %lf\n", Ener-deltaE, Ener);
        //printf("%lf, %lf\n", valprec, valatt);

        // Refinement

        zero = (Ener-deltaE) - ( deltaE / (valatt - valprec) ) * valatt;

        // Set restart
        if (n == l + 1) {
          restart = zero;
        }

        j += 1;

        // Exact energy of the 3D harmonic oscillator

        Eexact = (2.*n + l + (3. / 2.));

        printf("%.10lf, %.10lf\n", zero, Eexact);
        fprintf(fp, "%.10lf, %.10lf\n", zero, Eexact);

      }

      fclose(fp);

    }

    //system("gnuplot -p plot.gp");

    return 0;
}
