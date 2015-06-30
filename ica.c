#include <lapacke/lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "util/util.h"

double EPS = 1e-18;
double MAX_W = 1e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;

void pca_whiten(gsl_matrix *input,  size_t const NCOMP,
                gsl_matrix *x_white,
                gsl_matrix *white,
                gsl_matrix *dewhite,
                int demean){
  // demean input matrix
  if (demean){
    matrix_demean(input);
  }
  // Convariance Matrix
  gsl_matrix *cov = gsl_matrix_alloc(input->size1, input->size1);
  matrix_cov(input, cov);
  // Set up eigen decomposition
  gsl_vector *eval = gsl_vector_alloc(cov->size1); //eigen values
  gsl_matrix *evec = gsl_matrix_alloc(cov->size1, cov->size2); //eigen vector

  /*
  //Compute eigen values with LAPACK
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U',
    cov->size1, cov->data, cov->size1, eval->data);
  gsl_matrix_memcpy(evec,cov);
  */

  //Compute eigen values with GSL
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (cov->size1);
  gsl_eigen_symmv(cov, eval, evec, w);
  gsl_matrix_free(cov);
  gsl_eigen_symmv_free(w);

  // sort eigen values
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);
  // reduce number of components
  //Computing whitening matrix
  gsl_matrix_view temp = gsl_matrix_submatrix(evec, 0,0 , evec->size1, NCOMP);
  gsl_matrix_transpose_memcpy(white, &temp.matrix);
  gsl_vector_view v;
  double e;
  size_t i;
  // eval^{-1/2} evec^T
  for (i = 0; i < NCOMP; i++) {
    e = gsl_vector_get(eval,i);
    v = gsl_matrix_row(white,i);
    gsl_blas_dscal(1/sqrt(e), &v.vector);
  }
  // Computing dewhitening matrix
  gsl_matrix_memcpy(dewhite, &temp.matrix);
  // evec eval^{1/2}
  for (i = 0; i < NCOMP; i++) {
    e = gsl_vector_get(eval,i);
    v = gsl_matrix_column(dewhite,i);
    gsl_blas_dscal(sqrt(e), &v.vector);
  }
  // whitening data (white x Input)

  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,
    white, input, 0.0, x_white);
  gsl_matrix_free(evec);
  gsl_vector_free(eval);

}
/*
void w_update(double *unmixer, double *x_white, double *bias1,
              double *lrate1, double *error){

              }
void infomax1(double *x_white){

}
*/
