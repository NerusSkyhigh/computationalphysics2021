#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

 // Comment

 // Global variables

double deltaE = 0.00001; //Energy increment, set to 10^-5 for final simulation.
double Emin = 1.; // Potential minumum
double xmin = 0.; // x minumum
double deltax = 0.001; //delta x
int Nx = 5000; // Number of points in mesh
double m = 1.;
double hbar = 1.;

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
    int i, n, l;
    double rat, Eexact, restart=0;
    double Ener = Emin;
    double valatt=0., valprec=0.;
    double zero;
    int nmax = 2;
    char filename[20];
    FILE *fp;


    for(l=0; l <= nmax-1; l++){

      sprintf(filename, "Levels l=%i.csv", l); // make filename
      fp = fopen(filename, "w+"); //Open file
      fprintf(fp, "Levels, Exact\n"); // Print column title

      printf("Levels l=%i\n", l); //Print on screen l for visual check

      // Reset energy value for every l cycle

      Ener = restart;

      // Initial conditions for l even or odd

      if( l%2 == 1 ){

        psi[0] = 1;
        psi[1] = 1- deltax; // condition 2 fixed using the linearity of the solution in 0

      } else{

        psi[0] = 0;
        psi[1] = deltax;

      }

      for(n=0; n<=nmax; n++){

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

        zero = Ener - ( deltaE / (valatt - valprec) ) * valatt;

        /* Memorize the lowest level for this l, the next search will start from here,
        just to save time */

        if(n == 0){
          restart = zero;
        }

        // Exact energy of the 3D harmonic oscillator

        Eexact = (2 * n + l + (3. / 2.));

        printf("%lf, %lf\n", zero, Eexact);
        fprintf(fp, "%lf, %lf\n", zero, Eexact);

      }

      fclose(fp);

    }

    //system("gnuplot -p plot.gp");

    return 0;
}
