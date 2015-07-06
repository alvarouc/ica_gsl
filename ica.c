// #include <lapacke/lapacke.h>
#include "util/util.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>

double EPS = 1e-18;
double MAX_W = 1.0e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;
size_t MAX_STEP= 512;

double logit(double in){
  // NOTE: gsl_expm1 computes exp(x)-1, hence the 2 + in denominator
    return 1.0 - (2.0 / (2.0 + gsl_expm1(-in)));
}

void pca_whiten(
  gsl_matrix *input,// NOBS x NVOX
  size_t const NCOMP, //
  gsl_matrix *x_white, // NCOMP x NVOX
  gsl_matrix *white, // NCOMP x NCOMP
  gsl_matrix *dewhite, //NOBS x NVOX
  int demean){

  // get input reference
    size_t NSUB = input->size1;

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
  gsl_matrix_view temp = gsl_matrix_submatrix(evec, 0,0 , NSUB, NCOMP);
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

int w_update(
  gsl_matrix *weights,
  gsl_matrix *x_white,
  gsl_matrix *bias,
  double lrate){

  int error = 0;
  const size_t NVOX = x_white->size2;
  const size_t NCOMP = x_white->size1;
  size_t block = (size_t)floor(sqrt(NVOX/3.0));
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
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, rand());
  gsl_ran_shuffle(r, permute->data, permute->size, sizeof(size_t));
  // gsl_ran_shuffle(r, permute->data, NVOX, sizeof (size_t));

  size_t start;
  gsl_matrix *sub_x_white = gsl_matrix_alloc(NCOMP, block);
  gsl_matrix *unmixed     = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *unm_logit   = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *temp_I      = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *ones        = gsl_matrix_alloc(block,1);
  gsl_matrix_set_all(ones, 1.0);
  double max;

  gsl_matrix *d_unmixer = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_vector_view src, dest;
  for (start = 0; start < NVOX; start = start + block) {
    if (start + block > NVOX-1){
      block = NVOX-start;
      gsl_matrix_free(sub_x_white);
      sub_x_white= gsl_matrix_alloc(NCOMP, block);
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
    // sub_x_white = xwhite[:, permute[start:start+block]]
    for (i = start; i < start+block; i++) {
      src = gsl_matrix_column(x_white, gsl_vector_get(permute, i));
      dest = gsl_matrix_column(sub_x_white, i-start);
      gsl_vector_memcpy(&dest.vector, &src.vector);
    }
    // Compute unmixed = weights . sub_x_white + bias . ib
    matrix_mmul(weights, sub_x_white, unmixed);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
      1.0, bias, ib, 1.0, unmixed);
    // Compute 1-2*logit
    gsl_matrix_memcpy(unm_logit, unmixed);
    matrix_apply_all(unm_logit, logit);
    // weights = weights + lrate*(block*I+(unm_logit*unmixed.T))*weights
    gsl_matrix_set_identity(temp_I); // temp_I = I
    // (1) temp_I = block*temp_I +unm_logit*unmixed.T
    gsl_blas_dgemm( CblasNoTrans,CblasTrans,
                    1.0, unm_logit, unmixed,
                    (double)block , temp_I);
    // BE CAREFUL with aliasing here! use d_unmixer if problems arise
    gsl_matrix_memcpy(d_unmixer, weights);
    // (2) weights = weights + lrate*temp_I*weights
    gsl_blas_dgemm( CblasNoTrans,CblasNoTrans,
                    lrate, temp_I, d_unmixer,
                    1.0, weights);
    // Update the bias
    gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,
                    lrate, unm_logit, ones,
                    1.0,  bias);
    // check if blows up
    max = gsl_matrix_max(weights);
    if (max > MAX_W){

      if (lrate<1e-6) {
        printf("\nERROR: Weight matrix may not be invertible\n");
        error = 2;
        break;
      }
      error = 1;
      break;
    }
  }


  //clean up
  gsl_rng_free (r);
  gsl_matrix_free(d_unmixer);
  gsl_vector_free(permute);
  gsl_matrix_free(ib);
  gsl_matrix_free(unmixed);
  gsl_matrix_free(temp_I);
  gsl_matrix_free(sub_x_white);
  gsl_matrix_free(ones);
  gsl_matrix_free(unm_logit);
  return(error);

}

void infomax(gsl_matrix *x_white, gsl_matrix *A, gsl_matrix *S){
  /*Computes ICA infomax in whitened data
    Decomposes x_white as x_white=AS
    *Input
    x_white: whitened data (Use PCAwhiten)
    *Output
    A : mixing matrix
    S : source matrix
  */
  int verbose = 1; //true

  size_t NCOMP = x_white->size1;
  gsl_matrix *weights        = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *old_weights    = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *bias           = gsl_matrix_calloc(NCOMP, 1);
  gsl_matrix *weights_change = gsl_matrix_calloc(NCOMP,NCOMP);
  gsl_matrix *old_wt_change  = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *temp_change    = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix_set_identity(weights);
  gsl_matrix_set_identity(old_weights);
  double lrate = 0.005/log((double)NCOMP);
  double change;
  double angle_delta =0;
  size_t step = 0;
  int error = 0;
  while(step < MAX_STEP){
    error = w_update(weights, x_white, bias, lrate);
    if (error==1 || error==2){
      // It blowed up! RESTART!
      step = 0;
      // change = 1;
      error = 0;
      lrate *= ANNEAL;
      gsl_matrix_set_identity(weights);
      gsl_matrix_set_identity(old_weights);
      gsl_matrix_set_zero(old_wt_change);
      gsl_matrix_set_zero(bias);

      if (lrate > MIN_LRATE){
        printf("\nLowering learning rate to %g and starting again.\n",lrate);
      }
      else{
        printf("\nMatrix may not be invertible");
      }
    }
    else if (error==0){
      // WEIGHTS_CHANGE <- WEIGHTS - OLD_WEIGHTS
      gsl_matrix_memcpy(weights_change, weights);
      gsl_matrix_sub(weights_change, old_weights);
      // old_weights <- weights
      gsl_matrix_memcpy(old_weights, weights);
      change = matrix_norm(weights_change);
      step ++;
      if (step > 2){
        // Compute angle delta
        gsl_matrix_memcpy(temp_change, old_wt_change);
        gsl_matrix_mul_elements(temp_change, weights_change);
        angle_delta = acos(matrix_sum(temp_change) / sqrt(matrix_norm(weights_change)*(matrix_norm(old_wt_change))));
        angle_delta *= (180.0 / M_PI);
      }

      if (angle_delta > 60){
        lrate *= ANNEAL;
        gsl_matrix_memcpy(old_wt_change, weights_change);
        // old_change = change;

      } else if (step==1) {
        // old_change = change;
        gsl_matrix_memcpy(old_wt_change, weights_change);
      }

      if ((verbose && (step % 1)== 0) || change < W_STOP){
        printf("\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f",
          step, lrate, change, angle_delta);
      }


      if (change < W_STOP) step = MAX_STEP;
      // if (change > 1.0e3 ) lrate *= ANNEAL;
      }

  }


  // weights ^-1
  matrix_inv(weights, A);
  matrix_mmul(weights, x_white, S);
  gsl_matrix_free(old_wt_change);
  gsl_matrix_free(weights);
  gsl_matrix_free(old_weights);
  gsl_matrix_free(bias);
}

void ica(gsl_matrix *A, gsl_matrix *S, gsl_matrix *X){

  const size_t NCOMP = A->size2;
  const size_t NSUB = X->size1;
  const size_t NVOX = X->size2;
  gsl_matrix *white_A = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *white_X = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *white   = gsl_matrix_alloc(NCOMP, NSUB);
  gsl_matrix *dewhite = gsl_matrix_alloc(NSUB, NCOMP);
  printf("\nPCA decomposition ...");
  pca_whiten(X, NCOMP, white_X, white, dewhite, 1);
  printf("Done.");
  printf("\nINFOMAX ...");
  infomax(white_X, white_A, S);
  printf("Done");
  matrix_mmul(dewhite, white_A, A);

  gsl_matrix_free(white_A);
  gsl_matrix_free(white_X);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);

}

/*
void infomax1(double *x_white){

}
*/
