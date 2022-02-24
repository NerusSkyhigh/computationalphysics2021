#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>


int Np = 32; //number of particles
double sigma = 0.2556; //unit of length
double n0 = 21.86; //saturation density
double epsilon = 10.22; // unit of energy
double b = 1.23;

int main(void)
{
  double dr[Np][Np];
  double dx[Np][Np];
  double dy[Np][Np];
  double dz[Np][Np];
  double r[3][Np];
  double Ne = 10000;
  double delta = 0.11; //from some trials it seems the best value
  double l = 0.5 * cbrt(Np/n0) /sigma;
  double L = 2*l;
  //FILE *fp;
  //fp = fopen("potential.csv","w");
  double rprop[3][Np];
  double drprop[Np][Np];
  double dxprop[Np][Np];
  double dyprop[Np][Np];
  double dzprop[Np][Np];
  double eff;
  srand(time(0));// NEEDED TO GENERATE ALWAYS DIFFERENT RANDOM NUMBERS

// THERMALIZATION AND LOOKING FOR THE CORRECT DELTA TO GET A 50% PROBABILITY
do{
  // Initial position of each particle
  int n = 0;
  for (int k = 0; k<4; k++){
      for (int i = 0; i<4; i++){
          for (int j = 0; j<2; j++){
              r[2][n] = k * l / 2;  // set z coordinate
              r[0][n] = i * l / 2; // set x coordinate

              if (k%2 == 0){
                  r[1][n] = j * l + i%2 * l/2; // set y coordinate
              }
              else {
                  r[1][n] = (j + 1/2) * l - i%2 * l/2; // set y coordinate
              }
              n = n+1;
          }
      }
  }
  int acc = 0;
  int rej = 0;
  printf("%f\n", delta);

  for (int n=0; n<Ne; n++){
      printf("Equilibrium %d\n", n);
      // new positions proposal with attention to boundary conditions
    for (int i=0;i<3;i++){
      for (int j=0;j<Np;j++){
        double r1 = (double)rand()/RAND_MAX;
        //printf("%f\n", r1);
        rprop[i][j] = r[i][j] + delta*(r1-0.5);
        rprop[i][j] = rprop[i][j] - L * round(rprop[i][j]/L);
      }
    }
      // Calculation of all the possible distances in all directions for the 2 sets of positions
    for (int i=0;i<Np;i++){
      for (int j=0;j<Np;j++){
        dx[i][j] = r[0][i]-r[0][j];
        dx[i][j] = dx[i][j] - L* round(dx[i][j]/L);
        dy[i][j] = r[1][i]-r[1][j];
        dy[i][j] = dy[i][j] - L* round(dy[i][j]/L);
        dz[i][j] = r[2][i]-r[2][j];
        dz[i][j] = dz[i][j] - L* round(dz[i][j]/L);

        dr[i][j] = sqrt( pow(dx[i][j],2)+pow(dy[i][j],2)+pow(dz[i][j],2));

        dxprop[i][j] = rprop[0][i]-rprop[0][j];
        dxprop[i][j] = dxprop[i][j] - L* round(dxprop[i][j]/L);
        dyprop[i][j] = rprop[1][i]-rprop[1][j];
        dyprop[i][j] = dyprop[i][j] - L* round(dyprop[i][j]/L);
        dzprop[i][j] = rprop[2][i]-rprop[2][j];
        dzprop[i][j] = dzprop[i][j] - L* round(dzprop[i][j]/L);

        drprop[i][j] = sqrt( pow(dxprop[i][j],2)+pow(dyprop[i][j],2)+pow(dzprop[i][j],2));
        }
      }

      // Calculate probability
    double p = 0.0;
    double pprop = 0.0;
    for(int i=0; i<Np; i++){
        for(int j=0; j<i; j++){
            p += -pow((b/dr[i][j]), 5);
            pprop += -pow((b/drprop[i][j]), 5);
        }
    }


    double ratio = pow(M_E,pprop)/pow(M_E,p);
    //srand(time(0));
    double r2;
    for (int i=0;i<1;i++){
      r2 = (double)rand()/RAND_MAX;
    }
    //printf("%f, %f\n", p, pprop);

    if (ratio>r2){
      acc = acc+1;
      for (int i=0;i<Np;i++){
        for (int j=0;j<Np;j++){
          dr[i][j] = drprop[i][j];
          dx[i][j] = dxprop[i][j];
          dy[i][j] = dyprop[i][j];
          dz[i][j] = dzprop[i][j];
        }
      }
      for (int k=0;k<3;k++){
        for (int i=0;i<Np;i++){
          r[k][i] = rprop[k][i];
        }
      }
    }
    else{
      rej = rej +1;
    }
    //printf("%d\n", acc);
    //printf("%d\n", rej);
  }
  double Nsamples = acc + rej;
  //printf("%f\n", Nsamples);
  eff = acc / Nsamples;
  printf("%f\n", eff);
  //printf("%f\n",delta);
  delta = delta + 0.01;
}while (eff >0.6 || eff<0.4);


delta = delta -0.01;

// ACTUAL CALCULATION OF THE ENERGIES AND SAMPLING PROCUDURE
int acc = 0;
int rej = 0;
int Nsteps = 20000;
double V[Nsteps];
double T[Nsteps];
double E[Nsteps];
double Tjf[Nsteps];

double Esum = 0;
double Esum2 = 0;


for (int n=0; n<Nsteps; n++){
    printf("%d\n", n);
  // new positions proposal with attention to boundary conditions
  for (int i=0;i<3;i++){
    for (int j=0;j<Np;j++){
      double r1 = (double)rand()/RAND_MAX;
      //printf("%f\n", r1);
      rprop[i][j] = r[i][j] + delta*(r1-0.5);
      rprop[i][j] = rprop[i][j] - L * round(rprop[i][j]/L);
    }
  }
    // Calculation of all the possible distances in all directions for the 2 sets of positions
  for (int i=0;i<Np;i++){
    for (int j=0;j<Np;j++){
      dx[i][j] = r[0][i]-r[0][j];
      dx[i][j] = dx[i][j] - L* round(dx[i][j]/L);
      dy[i][j] = r[1][i]-r[1][j];
      dy[i][j] = dy[i][j] - L* round(dy[i][j]/L);
      dz[i][j] = r[2][i]-r[2][j];
      dz[i][j] = dz[i][j] - L* round(dz[i][j]/L);

      dr[i][j] = sqrt( pow(dx[i][j],2)+pow(dy[i][j],2)+pow(dz[i][j],2));

      dxprop[i][j] = rprop[0][i]-rprop[0][j];
      dxprop[i][j] = dxprop[i][j] - L* round(dxprop[i][j]/L);
      dyprop[i][j] = rprop[1][i]-rprop[1][j];
      dyprop[i][j] = dyprop[i][j] - L* round(dyprop[i][j]/L);
      dzprop[i][j] = rprop[2][i]-rprop[2][j];
      dzprop[i][j] = dzprop[i][j] - L* round(dzprop[i][j]/L);

      drprop[i][j] = sqrt( pow(dxprop[i][j],2)+pow(dyprop[i][j],2)+pow(dzprop[i][j],2));
      }
    }

    // Calculate probability
  double p = 0.0;
  double pprop = 0.0;
  for(int i=0; i<Np; i++){
      for(int j=0; j<i; j++){
          p += -pow((b/dr[i][j]), 5);
          pprop += -pow((b/drprop[i][j]), 5);
      }
  }


  double ratio = pow(M_E,pprop)/pow(M_E,p);
  //srand(time(0));
  double r2;
  for (int i=0;i<1;i++){
    r2 = (double)rand()/RAND_MAX;
  }
  //printf("%f, %f\n", p, pprop);

  if (ratio>r2){
    acc = acc+1;
    for (int i=0;i<Np;i++){
      for (int j=0;j<Np;j++){
        dr[i][j] = drprop[i][j];
        dx[i][j] = dxprop[i][j];
        dy[i][j] = dyprop[i][j];
        dz[i][j] = dzprop[i][j];
      }
    }
    for (int k=0;k<3;k++){
      for (int i=0;i<Np;i++){
        r[k][i] = rprop[k][i];
      }
    }
  }
  else{
    rej = rej +1;
  }

// Given the new configuration we calculate the value of the potential
V[n] = 0;
for (int i=0;i<Np;i++){
  for(int j=0; j<i;j++){
    V[n] += 4*(pow(1/dr[i][j],12)- pow(1/dr[i][j],6));
  }
}

// Given the new configuration we calculate the local kinteic energy
T[n] = 0;
Tjf[n] = 0;
double G = 0;

for (int l=0;l<Np;l++){
  for(int i=0; i<Np;i++){
    if (i!=l){
      T[n] += 10*pow(b,5)* pow(1/dr[l][i] , 7);

      for(int k=0; k<Np;k++){
        if(k!=l){
        G += -(25/4) * pow(b,10) * dx[l][i]*dx[l][k] / ( pow(dr[l][i] , 7) * pow(dr[l][k] , 7));
        G += -(25/4) * pow(b,10) * dy[l][i]*dy[l][k] / ( pow(dr[l][i] , 7) * pow(dr[l][k] , 7));
        G += -(25/4) * pow(b,10) * dz[l][i]*dz[l][k] / ( pow(dr[l][i] , 7) * pow(dr[l][k] , 7));
        }
      }
    }
  }
}
T[n] = 0.09094*(T[n]+G);
Tjf[n] = 0.09094*(T[n]/2);

E[n] = T[n]+V[n];

Esum += E[n];
Esum2 += pow(E[n],2);

}
double Nsamples = acc + rej;
printf("%f\n", Nsamples);
eff = acc / Nsamples;
printf("%f\n", eff);

//FILE *fp;
//fp=fopen("energies.csv","w+");
for (int n=0; n<Nsteps; n++){
  printf("%f, %f, %f \n", V[n],T[n],E[n]);
}

// We can now calculate the average value of the local energy and its error
double Evalue;
double deltaE;

Evalue = Esum/Nsteps;
deltaE = sqrt( 1/(Nsteps-1) * (Esum2/Nsteps - Evalue*Evalue) ); //for some reason it doesn't work
printf("value: %f error: %f \n", Evalue, deltaE);
printf("Esum: %f Esum2: %f \n", Esum, Esum2);

return 0;
}
