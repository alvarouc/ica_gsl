#include "util.h"

void matrix_demean(gsl_matrix *input){

  gsl_vector *mean = matrix_mean(input);

  size_t NCOL = input->size2;
  size_t i;
  for (i = 0; i < NCOL; i++) {
    gsl_vector_view column = gsl_matrix_column(input, i);
    gsl_vector_add_constant( &column.vector,
                             -gsl_vector_get(mean, i));
  }
}

gsl_vector *matrix_mean(gsl_matrix *input){
  //  Function to extract the column mean of a gsl matrix
  size_t col;
  size_t NCOL = input->size2;
  gsl_vector_view a_col;
  gsl_vector *mean = gsl_vector_alloc(NCOL) ;
  for (col = 0; col < NCOL; col++) {
    a_col = gsl_matrix_column(input, col);
    gsl_vector_set(mean, col, gsl_stats_mean(a_col.vector.data,
    a_col.vector.stride, a_col.vector.size));
  }

  return mean;

}

void fill_vector_const(gsl_vector *input, float const x){
  size_t N = input->size;
  size_t i;
  for (i = 0; i < N; i++)
    gsl_vector_set(input, i, x);

}

void fill_matrix_const( gsl_matrix *input, float const x){
  size_t NROW = input->size1;
  size_t NCOL = input->size2;

  int i,j;

  for (i = 0; i < NROW; i++)
    for (j = 0; j < NCOL; j++){
      gsl_matrix_set(input, i, j, x);
    }
}

void fill_matrix_random(gsl_matrix *input){
// Fill a GSL matrix with random numbers
  size_t NROW = input->size1;
  size_t NCOL = input->size2;

  int i,j;

  for (i = 0; i < NROW; i++)
    for (j = 0; j < NCOL; j++){
                  gsl_matrix_set(input, i, j, (double)(rand()%100));
          }

}

void print_matrix_corner(gsl_matrix *input){

  int i,j;

  printf ("\n\nMatrix size: %zux%zu\n", input->size1, input->size2);

  size_t NROW = input->size1 < 6 ? input->size1 : 6;
  size_t NCOL = input->size2 < 6 ? input->size2 : 6;

  for (i = 0; i < NROW; i++){
    for (j = 0;j < NCOL; j++) {
      printf("%g ", gsl_matrix_get(input,i,j));
    }
    printf("\n");
  }

}

void print_vector_head(gsl_vector *input){
  int i;

  printf ("\n\nVector size: %zu\n", input->size);

  size_t N = input->size < 10 ? input->size : 10;
  printf("[ ");
  for (i = 0; i < N; i++){
      printf("%g ", gsl_vector_get(input,i));
  }
  printf(" ... ]\n\n");

}

gsl_matrix *matrix_cov(gsl_matrix *input){
  /*Compute matrix covariance
  The input is a matrix with an observation per row
  The function assumes the matrix is demeaned
  Note: This can be optimized to exploid matrix symmetry
  */

  gsl_matrix *cov = gsl_matrix_alloc(input->size1, input->size1);

  gsl_blas_dgemm (CblasNoTrans, CblasTrans,
    1.0, input, input, 0.0, cov);
  gsl_matrix_scale(cov, 1.0/input->size1);
  return cov;
}
