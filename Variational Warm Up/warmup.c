#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include<gsl/gsl_blas.h>

/* Funzione per il calcolo della matrice di overlap */

double overlap(int n, int m)
{
  double s, num, den;

  num = 8. * ( 1. + pow(-1., n + m) );
  den = (1. + m + n) * (3. + m + n) * (5. + m + n);

  s = num / den;

  return s;
}

// FUNCTION FOR H MATRIX

double hmatrix(int n, int m)
{
  double h, num, den;

  if( n + m > 1)
  {
    num = 4. * (1. + pow(-1., m + n)) * (-1. + m + n + 2. * m * n);
    den = (-1. + m + n) * (1. + m + n) * (3. + m + n);
    h = num / den;
  } else if (n + m <= 1)
  {
    h = 0;
  }

  return h;
}

// FUNZIONE MAIN

int main ()
{
  int dim = 4;/*Dimension of the matrix */

  /* The first step is the construction of the overlap matrix*/

  gsl_matrix *S = gsl_matrix_alloc(dim,dim); /*Allocation of the matrix*/

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      gsl_matrix_set(S, i, j, overlap(i,j));
    }
  }

  /* Print overlap matrix*/

  printf("\n OVERLAP MATRIX \n");

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      printf ("S(%d,%d) = %g\n", i, j, gsl_matrix_get (S, i, j));


  gsl_vector *eval = gsl_vector_alloc(dim); /* eigenvalue vector*/
  gsl_matrix *evec = gsl_matrix_alloc(dim, dim); /*eigenvectors matrix U*/
  gsl_vector *evec_i = gsl_vector_alloc(dim); // vector

  gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(dim);

  gsl_eigen_symmv(S, eval, evec, w); //compute eigenvalues and eigenvectors

  gsl_eigen_symmv_free(w); //free workspace

  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC); //sort eigenvalues

  gsl_matrix_free (S); // Free matrix S

  /* Building of matrix V */

  gsl_matrix *V = gsl_matrix_alloc(dim,dim); // Allocate matrix

  printf("\n EIGENVALUES AND EIGENVECTORS OF OVERLAP MATRIX \n");

  for (int i = 0; i < dim; i++)
    {
      double eval_i = gsl_vector_get(eval, i);
      gsl_vector_view evec_pl = gsl_matrix_column(evec, i);

      /* Just plotting eigenvalues and vectors of S here*/

      printf("\neigenvalue = %g", eval_i);
      printf("\n eigenvector: \n");
      gsl_vector_fprintf(stdout, &evec_pl.vector, "%g");

      /* Building V matrix*/

      for (int j = 0; j < dim; j++)
      {
          gsl_matrix_set(V, j, i, (1 / sqrt(eval_i)) * gsl_matrix_get(evec,j,i));
      }

    }

    printf("\n MATRIX V \n");

    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        printf ("V(%d,%d) = %g\n", i, j, gsl_matrix_get (V, i, j));

  // Free the memory
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  gsl_vector_free(evec_i);

  // MATRIX H

  gsl_matrix *H = gsl_matrix_alloc(dim,dim); /*Allocation of the matrix*/

  // Building the matrix using the function
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      gsl_matrix_set(H, i, j, hmatrix(i,j));
    }
  }

  /* Print H matrix*/

  printf("\n H MATRIX \n");

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      printf ("H(%d,%d) = %g\n", i, j, gsl_matrix_get (H, i, j));

  /* BUILDING THE H' MATRIX */

  gsl_matrix *Vtr = gsl_matrix_alloc(dim,dim); /*Alloc transpose matrix*/

  // Build the transpose

  gsl_matrix_transpose_memcpy(Vtr, V);

  /* Multiplication Vtr and H. For this we need the blas routine (either dgemm
  or sgemm according to where you look). Here i used https://sites.google.com/a/phys.buruniv.ac.in/numerical/laboratory/example-codes/matrix-vector-operations-with-gsl-blas */

  gsl_matrix *VtrH = gsl_matrix_alloc(dim,dim); /*Alloc result matrix*/

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Vtr, H, 0.0, VtrH);

  //Free memory
  gsl_matrix_free(H);

  // Now multiply with V

  gsl_matrix *Hprime = gsl_matrix_alloc(dim,dim); /*Alloc result matrix*/

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, VtrH, V, 0.0, Hprime);

  // Free memory
  gsl_matrix_free(V);
  gsl_matrix_free(VtrH);

  //Print H'

  printf("\n H' MATRIX \n");

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      printf ("H'(%d,%d) = %g\n", i, j, gsl_matrix_get (Hprime, i, j));


  // Find Eigenvalues and Eigenvectors of H'

  gsl_vector *evalf = gsl_vector_alloc(dim); /* eigenvalue vector*/
  gsl_matrix *evecf = gsl_matrix_alloc(dim, dim); /*eigenvectors matrix U*/

  gsl_eigen_symmv_workspace * wf = gsl_eigen_symmv_alloc(dim);

  gsl_eigen_symmv(Hprime, evalf, evecf, wf); //compute eigenvalues and eigenvectors

  gsl_eigen_symmv_free(wf); //free workspace

  gsl_eigen_symmv_sort(evalf, evecf, GSL_EIGEN_SORT_ABS_ASC); //sort eigenvalues

  gsl_matrix_free (Hprime); // Free matrix S

  /* Printing the final results */

  printf("\n EIGENVALUES AND EIGENVECTORS OF H' MATRIX \n");

  for (int i = 0; i < dim; i++)
    {
      double eval_i = gsl_vector_get(evalf, i);
      gsl_vector_view evec_pl = gsl_matrix_column(evecf, i);

      /* Just plotting eigenvalues and vectors of S here*/

      printf("\neigenvalue = %g", eval_i);
      printf("\n eigenvector: \n");
      gsl_vector_fprintf(stdout, &evec_pl.vector, "%g");

    }

    // Free the memory
    gsl_vector_free(evalf);
    gsl_matrix_free(evecf);

  return 0;
}
