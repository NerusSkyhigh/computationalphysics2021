#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

double j(int l, double x)
{
  // Round zero to the closes non zero double
  // This avoid division by zero but give a
  // good approximation for the limit
  if(x==0)
    x = DBL_TRUE_MIN;

  if(l < -1) {
    printf("l should be >= 0\n");
    exit(-1);
  }

  if(l==-1) {
    return cos(x)/x;
  } else if(l==0) {
    return sin(x)/x;
  } else {
    return (2*(l-1)+1)/x * j(l-1, x) - j(l-2, x);
  }
}

double n(int l, double x)
{
  // Round zero to the closes non zero double
  // This avoid division by zero but give a
  // good approximation for the limit
  if(x==0)
    x = DBL_TRUE_MIN;

  if(l < -1) {
    printf("l should be >= 0\n");
    exit(-1);
  }

  if(l==-1) {
    return sin(x)/x;
  } else if(l==0) {
    return -1*cos(x)/x;
  } else {
    return (2*(l-1)+1)/x * n(l-1, x) - n(l-2, x);
  }
}

int main (int argc, char** argv) {
  /*
     We use the main function as a test routine
  */
  double x = 0; // initial point
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
