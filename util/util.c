#include "util.h"
#include <gsl/gsl_math.h>
#include <math.h>
// #include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <omp.h>
#include <lapacke.h>

double absolute(double value) {
  if (value < 0) {
    return -value;
  }
  else {
    return value;
  }
}

void gsl_eig(gsl_matrix *sym, gsl_vector *eval, gsl_matrix *evec ,size_t NCOMP){
  //Compute eigen values with GSL
  size_t NSUB = sym->size1;
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(NSUB);
  gsl_eigen_symmv(sym, eval, evec, w);
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);
  gsl_matrix_view temp = gsl_matrix_submatrix(sym, 0,0 , NSUB, NCOMP);
  gsl_matrix_memcpy(evec,&temp.matrix);
}

void si_eig(gsl_matrix *sym, gsl_vector *eval, gsl_matrix *evec ,size_t NCOMP){
  // simple Eigen decomposition
  // gsl_matrix *evec = gsl_matrix_alloc(NSUB, NSUB);
  // gsl_vector *eval = gsl_vector_alloc(NCOMP); //eigen values
  size_t NSUB = sym->size1;
  gsl_vector *eval_temp =gsl_vector_alloc(NSUB);
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U',
    NSUB, sym->data, NSUB, eval_temp->data);
  gsl_eigen_symmv_sort (eval_temp, sym, GSL_EIGEN_SORT_ABS_DESC);
  gsl_matrix_view temp = gsl_matrix_submatrix(sym, 0,0 , NSUB, NCOMP);
  gsl_matrix_memcpy(evec,&temp.matrix);
  gsl_vector_view temp_vec = gsl_vector_subvector(eval_temp, 0, NCOMP);
  gsl_vector_memcpy(eval, &temp_vec.vector);
  gsl_vector_free(eval_temp);
}

void dc_eig(gsl_matrix *sym, gsl_vector *eval, gsl_matrix *evec ,size_t NCOMP){
  // Divide and conquer Eigen decomposition
  // gsl_matrix *evec = gsl_matrix_alloc(NSUB, NSUB);
  // gsl_vector *eval = gsl_vector_alloc(NCOMP); //eigen values
  size_t NSUB = sym->size1;
  gsl_vector *eval_temp =gsl_vector_alloc(NSUB);
  LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U',
    NSUB, sym->data, NSUB, eval_temp->data);
  gsl_eigen_symmv_sort (eval_temp, sym, GSL_EIGEN_SORT_ABS_DESC);
  gsl_matrix_view temp = gsl_matrix_submatrix(sym, 0,0 , NSUB, NCOMP);
  gsl_matrix_memcpy(evec,&temp.matrix);
  gsl_vector_view temp_vec = gsl_vector_subvector(eval_temp, 0, NCOMP);
  gsl_vector_memcpy(eval, &temp_vec.vector);
  gsl_vector_free(eval_temp);
}

void rr_eig(gsl_matrix *sym, gsl_vector *eval, gsl_matrix *evec, size_t NCOMP ){
  // Relative robust Eigen decomposition
  size_t NSUB = sym->size1;
  lapack_int m=0;
  double abstol=-1.0, vl=0.0, vu=0.0;
  lapack_int *ifail = (lapack_int *)LAPACKE_malloc( sizeof(lapack_int) * NSUB );
  LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
    NSUB, sym->data, NSUB, vl, vu, NSUB-NCOMP+1, NSUB,
    abstol, &m, eval->data, evec->data, NCOMP, ifail);
}

void xx_eig(gsl_matrix *sym, gsl_vector *eval, gsl_matrix *evec, size_t NCOMP ){
  // simple Eigen decomposition with embedded cut
  size_t NSUB = sym->size1;
  lapack_int m=0;
  double abstol=-1.0, vl=0.0, vu=0.0;
  lapack_int *ifail = (lapack_int *)LAPACKE_malloc( sizeof(lapack_int) * NSUB );
  LAPACKE_dsyevx(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
    NSUB, sym->data, NSUB, vl, vu, NSUB-NCOMP+1, NSUB,
    abstol, &m, eval->data, evec->data, NCOMP, ifail);
}

void ica_match_gt(gsl_matrix *true_a, gsl_matrix *true_s,
  gsl_matrix *esti_a, gsl_matrix *esti_s){
  /* Sort estimated loading and source matrices to match
  ground truth*/
  const size_t NCOMP = true_s->size1;
  const size_t NVOX = true_s->size2;
  const size_t NSUB = true_a->size1;

  gsl_matrix *cs = gsl_matrix_alloc(NCOMP, NCOMP);
  // cs <- CORR(S, S')
  matrix_cross_corr_row(cs, true_s, esti_s);
  matrix_apply_all(cs, absolute);
  // index <- cs.max(axis = 1 );
  size_t i;
  gsl_vector_view a_row, b_row;
  gsl_vector *index = gsl_vector_alloc(NCOMP);
  for (i = 0; i < NCOMP; i++) {
    a_row = gsl_matrix_row(cs, i);
    gsl_vector_set(index, i,
      gsl_stats_max_index(a_row.vector.data,
                          a_row.vector.stride,
                          a_row.vector.size));
  }
  // Sort estimated sources
  // S' <- S'[index,:]
  gsl_matrix *temp = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix_memcpy(temp, esti_s);
  #pragma omp parallel for private(i,a_row,b_row)
  for (i = 0; i < NCOMP; i++) {
    a_row = gsl_matrix_row(esti_s, i);
    b_row = gsl_matrix_row(temp, gsl_vector_get(index, i));
    gsl_vector_memcpy(&a_row.vector, &b_row.vector);
  }
  gsl_matrix_free(temp);
  // Sort estimated loadings
  // A' <- A'[:,index]
  temp = gsl_matrix_alloc(NSUB, NCOMP);
  gsl_matrix_memcpy(temp, esti_a);

  #pragma omp parallel for private(i,a_row,b_row)
  for (i = 0; i < NCOMP; i++) {
    a_row = gsl_matrix_column(esti_a, i);
    b_row = gsl_matrix_column(temp, gsl_vector_get(index, i));

    gsl_vector_memcpy(&a_row.vector, &b_row.vector);
  }

  gsl_matrix_free(temp);
  gsl_matrix_free(cs);
  gsl_vector_free(index);

}

void matrix_cross_corr_row(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B){
  size_t i,j;
  gsl_vector_view a, b;
  double c;
  #pragma omp parallel for private(i,j,a,b,c)
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
  size_t i,j;
  gsl_vector_view a, b;
  double c;
  #pragma omp parallel for private(i,j,a,b,c)
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
  #pragma omp parallel for collapse(2) private(i,j)
  for (i = 0; i < input->size1; i++) {
    for (j = 0; j < input->size2; j++) {
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
  #pragma omp parallel for private(i,j) reduction(+:accum)
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
  #pragma omp parallel for private(i,j) reduction(+:accum)
  for (i = 0; i < input->size1; i++) {
    for (j = 0; j < input->size2; j++) {
      accum += gsl_matrix_get(input,i,j);
    }
  }
  return (accum);

}

void matrix_demean(gsl_matrix *input){

  gsl_vector *mean = gsl_vector_alloc(input->size2);
  matrix_mean(mean, input);

  size_t NCOL = input->size2;
  size_t i;
  gsl_vector_view column;
  #pragma omp parallel for private(column)
  for (i = 0; i < NCOL; i++) {
    column = gsl_matrix_column(input, i);
    gsl_vector_add_constant( &column.vector,
                             -gsl_vector_get(mean, i));
  }
}

void matrix_mean(gsl_vector *mean, gsl_matrix *input){
  //  Function to extract the column mean of a gsl matrix
  size_t col;
  size_t NCOL = input->size2;
  gsl_vector_view a_col;
  #pragma omp parallel for private( a_col)
  for (col = 0; col < NCOL; col++) {
    a_col = gsl_matrix_column(input, col);
    gsl_vector_set(mean, col, gsl_stats_mean(a_col.vector.data,
    a_col.vector.stride, a_col.vector.size));
  }

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
  printf(" ... \n");
  // print last ten
  if (input->size>20){
      for (i = 10; i > 0; i--) {
        printf("%g ", gsl_vector_get(input, input->size-i));
      }
      printf("]\n\n");
  }

}

void matrix_cov(gsl_matrix *input, gsl_matrix *cov){
  /*Compute matrix covariance
  The input is a matrix with an observation per row
  The function assumes the matrix is demeaned
  Note: This can be optimized to exploid matrix symmetry
  */

  gsl_blas_dgemm (CblasNoTrans, CblasTrans,
    1.0, input, input, 0.0, cov);
  gsl_matrix_scale(cov, 1.0/(double)(input->size2));
}
