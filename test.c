#include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <CUnit/CUnit.h>
#include "ica.h"

// unit testing for ICA

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */
  // Input matrix
  size_t nrow = 100, ncol = 1000;
  gsl_matrix *input = gsl_matrix_alloc(nrow, ncol);

  CU_ASSERT_PTR_NOT_NULL(input)
}


void test_mmul()
{
  // dimensions of input matrix
  size_t m = 10000;
  size_t n = 10000;
  // number of components to extract
  int n_comp = 10;
  //allocating space for input
  gsl_matrix *input, *output;

  input = gsl_matrix_alloc(m,n);
  printf("Allocated matrix of %d by %d\n\n",m,n);
  //input initialization
  printf (" Intializing matrix data \n\n");

  for (i = 0; i < m; i++)
          for (j = 0; j < n; j++)
          {
                  gsl_matrix_set(input, i, j, (double)(rand()));
          }

  //output allocation
  output = gsl_matrix_alloc(m,n);
  // matrix multiplication
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,
                 1.0, input, input,
                 0.0, output);


}


// functional testing for ICA

int main(int argc, char const *argv[]) {

  printf("Unit Testing\n\n")

  printf("")
  /* code */
  return 0;
}
