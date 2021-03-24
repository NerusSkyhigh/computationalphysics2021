#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double j(int l, double x)
{
  if(l==-1)
  {
    return cos(x)/x;
  } else if(l==0) {
    return sin(x)/x;
  } else {
    return (2*(l-1)+1)/x * j(l-1, x) - j(l-2, x);
  }
  printf("Errore, la fuzione j(int n, double x) non dovrebbe arrivare qui");
}

double n(int l, double x)
{
  if(l==-1)
  {
    return sin(x)/x;
  } else if(l==0) {
    return -1*cos(x)/x;
  } else {
    return (2*(l-1)+1)/x * n(l-1, x) - n(l-2, x);
  }
  printf("Errore, la fuzione n(int n, double x) non dovrebbe arrivare qui");
}

int main (int argc, char** argv) {
  /*
     We use the main function as a test routine
  */
  double x = 1; // initial point
  double dx = 0.01; // increment
  int l = atoi(argv[2]);

  if(argc != 3) {
    printf("Invalid number of arguments\n");
    exit(-1);
  }
  //printf("x\t%s(%d, x)\n", argv[1], l);
  for(; x <= 20; x+=dx)
  {
    if(*argv[1] == 'j')
      printf("%2.6f\t%2.6f\n", x, j(l, x));

    if(*argv[1] == 'n')
      printf("%2.6f\t%2.6f\n", x, n(l, x));
  }
}
