#include "util.h"
#include <gsl/gsl_math.h>
#include <math.h>
// #include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>

void matrix_cross_corr_row(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B){
  /* NOTE: Paralelize the inner loop
  do instead
  for (i = 0; i < A->size2; i++) {
    for (j = i; j < B->size2; j++) {
  this reduces number of operations to half
  */
  size_t i,j;
  gsl_vector_view a, b;
  double c;
  for (i = 0; i < A->size1; i++) {
    for (j = 0; j < B->size1; j++) {
      a = gsl_matrix_row(A, i);
      b = gsl_matrix_row(B, j);
      c = gsl_stats_correlation(a.vector.data, a.vector.stride, b.vector.data, b.vector.stride, a.vector.size);
      gsl_matrix_set(C, i,j, c);
    }
  }


}

void matrix_cross_corr(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B){
  /* NOTE: Paralelize the inner loop
  do instead
  for (i = 0; i < A->size2; i++) {
    for (j = i; j < B->size2; j++) {
  this reduces number of operations to half
  */
  size_t i,j;
  gsl_vector_view a, b;
  double c;
  for (i = 0; i < A->size2; i++) {
    for (j = 0; j < B->size2; j++) {
      a = gsl_matrix_column(A, i);
      b = gsl_matrix_column(B, j);
      c = gsl_stats_correlation(a.vector.data, a.vector.stride, b.vector.data, b.vector.stride, a.vector.size);
      gsl_matrix_set(C, i,j, c);
    }
  }


}

void matrix_inv(gsl_matrix *input, gsl_matrix *output){

  int s;
  gsl_permutation * p = gsl_permutation_alloc (input->size1);
  gsl_linalg_LU_decomp (input, p, &s);
  gsl_linalg_LU_invert (input, p, output);
  gsl_permutation_free(p);

}

void random_matrix(gsl_matrix *input, double parameter,double (* func)(const gsl_rng *, double )){

  gsl_vector *temp = gsl_vector_alloc(input->size1 * input->size2);
  random_vector(temp, parameter, func);
  gsl_matrix_view temp_view = gsl_matrix_view_array(temp->data, input->size1, input->size2);
  gsl_matrix_memcpy(input, &temp_view.matrix);
  gsl_vector_free(temp);
}

void random_vector(gsl_vector *vec, double parameter , double (* func)(const gsl_rng *, double )){
  //
  const gsl_rng_type * T;
  gsl_rng * r;

  int i;

  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, rand());
  for (i = 0; i < vec->size; i++)
    {
      // double u = gsl_rng_uniform (r);
      gsl_vector_set(vec, i, func(r, parameter));
    }

  gsl_rng_free (r);
}

void matrix_apply_all(gsl_matrix *input, double (*fun)(double)){
  size_t i,j;
  for (j = 0; j < input->size2; j++) {
    for (i = 0; i < input->size1; i++) {
      gsl_matrix_set(input, i,j, fun(gsl_matrix_get(input, i,j)));
    }
  }

}


void matrix_mmul(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
  //  Computes C = A x B

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);

}

double matrix_norm(gsl_matrix *input){

  size_t i,j;
  double accum=0;
  for (i = 0; i < input->size1; i++) {
    for (j = 0; j < input->size2; j++) {
      accum += gsl_pow_2(gsl_matrix_get(input,i,j));
    }
  }

  return (accum);

}

double matrix_sum(gsl_matrix *input){

  size_t i,j;
  double accum=0;
  for (i = 0; i < input->size1; i++) {
    for (j = 0; j < input->size2; j++) {
      accum += gsl_matrix_get(input,i,j);
    }
  }
  return (accum);

}

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

void print_matrix_corner(gsl_matrix *input){

  int i,j;

  printf ("\nMatrix size: %zux%zu\n", input->size1, input->size2);

  size_t NROW = input->size1 < 6 ? input->size1 : 6;
  size_t NCOL = input->size2 < 6 ? input->size2 : 6;

  for (i = 0; i < NROW; i++){
    for (j = 0;j < NCOL; j++) {
      printf("%5.2f ", gsl_matrix_get(input,i,j));
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

void matrix_cov(const gsl_matrix *input, gsl_matrix *cov){
  /*Compute matrix covariance
  The input is a matrix with an observation per row
  The function assumes the matrix is demeaned
  Note: This can be optimized to exploid matrix symmetry
  */

  gsl_blas_dgemm (CblasNoTrans, CblasTrans,
    1.0, input, input, 0.0, cov);
  gsl_matrix_scale(cov, 1.0/(double)(input->size2));
  /*
  size_t i = 0;
  gsl_vector_view row;
  for (i = 0; i < cov->size2; i++) {
    row = gsl_matrix_row(cov,i);
    gsl_blas_dscal(1.0/(double)(input->size2), &row.vector);
  }*/
}
