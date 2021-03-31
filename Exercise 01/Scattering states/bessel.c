#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "bessel.h"

double besselj(int l, double x)
{
  /* Round zero to the closes non zero double */

  if(x==0)
    x = DBL_MIN;

  if(l < -1) {
    printf("l should be >= 0\n");
    exit(-1);
  }

  if(l==-1) {
    return cos(x)/x;
  } else if(l==0) {
    return sin(x)/x;
  } else {
    return (2*(l-1)+1)/x * besselj(l-1, x) - besselj(l-2, x);
  }
}

double besseln(int l, double x)
{
  /* Round zero to the closes non zero double */

  if(x==0)
    x = DBL_MIN;

  if(l < -1) {
    printf("l should be >= 0\n");
    exit(-1);
  }

  if(l==-1) {
    return sin(x)/x;
  } else if(l==0) {
    return -1*cos(x)/x;
  } else {
    return (2*(l-1)+1)/x * besseln(l-1, x) - besseln(l-2, x);
  }
}
