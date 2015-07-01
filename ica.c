#include <lapacke/lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "util/util.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>

double EPS = 1e-18;
double MAX_W = 1e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;

double logit(double in){
  return 1.0- 2.0*(1.0/(1.0 + exp(-in)));
}

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

void w_update(gsl_matrix *unmixer, gsl_matrix *x_white,
  gsl_matrix *bias, double *lrate, int *error)
{
  const size_t NVOX = x_white->size2;
  const size_t NCOMP = x_white->size1;
  size_t block = (size_t)floor(sqrt(NVOX/3.0));
  printf("\n***block size: %zu\n",block);
  gsl_matrix *ib = gsl_matrix_alloc(1,block);
  gsl_matrix_set_all( ib, 1.0);
  //getting permutation vector
  gsl_vector *permute = gsl_vector_alloc(NVOX);
  size_t i;
  for (i = 0; i < NVOX; i++) {
    gsl_vector_set(permute, i, i);
  }
  gsl_rng * r;
  const gsl_rng_type * T;
  // gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_ran_shuffle(r, permute->data, NVOX, sizeof (size_t));

  size_t start;
  gsl_matrix *sub_x_white =gsl_matrix_alloc(NCOMP, block);
  gsl_matrix *unmixed = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *unm_logit = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *temp_I = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *ones = gsl_matrix_alloc(block,1);
  gsl_matrix_set_all(ones, 1.0);

  // gsl_matrix *d_unmixer = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_vector_view src, dest;
  for (start = 0; start < NVOX; start = start + block) {
    if (start + block > NVOX-1){
      block = NVOX-start;
      gsl_matrix_free(sub_x_white);
      gsl_matrix_alloc(NCOMP, block);
      gsl_matrix_free(ib);
      ib = gsl_matrix_alloc(1,block);
      gsl_matrix_set_all( ib, 1.0);
      gsl_matrix_free(unmixed);
      unmixed = gsl_matrix_alloc(NCOMP,block);
      gsl_matrix_free(unm_logit);
      unm_logit = gsl_matrix_alloc(NCOMP,block);
      gsl_matrix_free(ones);
      ones = gsl_matrix_alloc(block,1);
      gsl_matrix_set_all(ones, 1.0);

    }

    for (i = start; i < start+block; i++) {
      src = gsl_matrix_column(x_white, gsl_vector_get(permute, i));
      dest = gsl_matrix_column(sub_x_white, i-start);
      gsl_vector_memcpy(&dest.vector, &src.vector);
    }
    // Compute unmixed = unmixer . sub_x_white + bias . ib
    matrix_mmul(unmixer, sub_x_white, unmixed);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
      1.0, bias, ib, 1.0, unmixed);
    // Compute 1-2*logit
    gsl_matrix_memcpy(unm_logit, unmixed);
    matrix_apply_all(unm_logit, logit);
    // unmixer = unmixer + lrate*block*I+(1-2*unmixed)
    // print_matrix_corner(sub_x_white);
    gsl_matrix_set_identity(temp_I);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,
    1.0, unm_logit, unmixed, (double)block , temp_I);
    // BE CAREFUL with aliasing here! use d_unmixer if problems arise
    // gsl_matrix_memcpy(d_unmixer, unmixer);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,
      *lrate, temp_I, unmixer, 1.0, unmixer);
    // Update the bias
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, *lrate,
      unm_logit, ones, 1.0,  bias);
    print_matrix_corner(bias);
    // check if blows up
    printf("max = %.2f",gsl_matrix_max(unmixer));

  }


  //clean up
  gsl_rng_free (r);
  gsl_vector_free(permute);
  gsl_matrix_free(ib);
  gsl_matrix_free(unmixed);
  gsl_matrix_free(temp_I);
  gsl_matrix_free(sub_x_white);
  gsl_matrix_free(ones);

}
/*
void infomax1(double *x_white){

}
*/
