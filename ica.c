// #include <lapack/lapacke.h>
#include <lapacke_utils.h>
#include "util/util.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_permute_vector.h>

double EPS = 1e-18;
double MAX_W = 1.0e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;
size_t MAX_STEP= 512;

double logit(double in){
  // NOTE: gsl_expm1 computes exp(x)-1, hence the 2 + in denominator
    // return 1.0 - (2.0 / (2.0 + gsl_expm1(-in)));
    return 1-(2 / (1 + exp(-in)) );
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

  //Compute eigen values with LAPACK
  LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U',
    cov->size1, cov->data, cov->size1, eval->data);
  gsl_matrix_memcpy(evec,cov);


  //Compute eigen values with GSL
  // gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (cov->size1 );
  // gsl_eigen_symmv(cov, eval, evec, w);
  // gsl_matrix_free(cov);
  // gsl_eigen_symmv_free(w);

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
  // #pragma omp parallel for private(i,e,v)
  for (i = 0; i < NCOMP; i++) {
    e = gsl_vector_get(eval,i);
    v = gsl_matrix_row(white,i);
    gsl_blas_dscal(1/sqrt(e), &v.vector);
  }
  // Computing dewhitening matrix
  gsl_matrix_memcpy(dewhite, &temp.matrix);

  // evec eval^{1/2}
  // #pragma omp parallel for private(i,e,v)
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
  size_t i;
  const size_t NVOX = x_white->size2;
  const size_t NCOMP = x_white->size1;
  size_t block = (size_t)floor(sqrt(NVOX/3.0));
  gsl_matrix *ib = gsl_matrix_alloc(1,block);
  gsl_matrix_set_all( ib, 1.0);
  //getting permutation vector

  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_permutation * p = gsl_permutation_alloc (NVOX);
  // gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_permutation_init (p);
  gsl_ran_shuffle (r, p->data, NVOX, sizeof(size_t));
  gsl_matrix *shuffled_x_white = gsl_matrix_alloc(NCOMP,NVOX);
  gsl_matrix_memcpy(shuffled_x_white, x_white);
  gsl_vector_view arow;
  // #pragma omp parallel for private(i,arow)
  for (i = 0; i < x_white->size1; i++) {
    arow = gsl_matrix_row(shuffled_x_white,i);
    gsl_permute_vector (p, &arow.vector);

  }

  size_t start;
  gsl_matrix *unmixed     = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *unm_logit   = gsl_matrix_alloc(NCOMP,block);
  gsl_matrix *temp_I      = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *ones        = gsl_matrix_alloc(block,1);
  gsl_matrix_set_all(ones, 1.0);
  double max;
  gsl_matrix_view sub_x_white_view;
  gsl_matrix *d_unmixer = gsl_matrix_alloc(NCOMP,NCOMP);
  for (start = 0; start < NVOX; start = start + block) {
    if (start + block > NVOX-1){
      block = NVOX-start;
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
    sub_x_white_view = gsl_matrix_submatrix(shuffled_x_white, 0,start, NCOMP, block );
    // Compute unmixed = weights . sub_x_white + bias . ib
    matrix_mmul(weights, &sub_x_white_view.matrix, unmixed); //put OPENBLAS_NUM_THREADS to maximum here
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
  // set number of threads back to normal
  // openblas_set_num _threads(MAX_THREAD);

  //clean up
  gsl_rng_free (r);
  gsl_permutation_free (p);
  gsl_matrix_free(d_unmixer);
  gsl_matrix_free(ib);
  gsl_matrix_free(unmixed);
  gsl_matrix_free(temp_I);
  gsl_matrix_free(ones);
  gsl_matrix_free(unm_logit);
  gsl_matrix_free(shuffled_x_white);
  return(error);

}

void infomax(gsl_matrix *x_white, gsl_matrix *weights, gsl_matrix *S, int  verbose){
  /*Computes ICA infomax in whitened data
    Decomposes x_white as x_white=AS
    *Input
    x_white: whitened data (Use PCAwhiten)
    *Output
    A : mixing matrix
    S : source matrix
  */
  // int verbose = 1; //true

  size_t NCOMP = x_white->size1;
  gsl_matrix *old_weights    = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *bias           = gsl_matrix_calloc(NCOMP, 1);
  gsl_matrix *d_weights      = gsl_matrix_calloc(NCOMP,NCOMP);
  gsl_matrix *temp_change    = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *old_d_weights  = gsl_matrix_calloc(NCOMP,NCOMP);

  gsl_matrix_set_identity(weights);
  gsl_matrix_set_identity(old_weights);
  double lrate = 0.005/log((double)NCOMP);
  double change=1;
  double angle_delta =0;
  size_t step = 1;
  int error = 0;
  while( (step < MAX_STEP) && (change > W_STOP)){
    error = w_update(weights, x_white, bias, lrate);
    if (error==1 || error==2){
      // It blowed up! RESTART!
      step = 1;
      // change = 1;
      error = 0;
      lrate *= ANNEAL;
      gsl_matrix_set_identity(weights);
      gsl_matrix_set_identity(old_weights);
      gsl_matrix_set_zero(d_weights);
      gsl_matrix_set_zero(old_d_weights);
      gsl_matrix_set_zero(bias);

      if (lrate > MIN_LRATE){
        printf("\nLowering learning rate to %g and starting again.\n",lrate);
      }
      else{
        printf("\nMatrix may not be invertible");
      }
    }
    else if (error==0){
      gsl_matrix_memcpy(d_weights, weights);
      gsl_matrix_sub(d_weights, old_weights);
      change = matrix_norm(d_weights);

      if (step > 2){
        // Compute angle delta
        gsl_matrix_memcpy(temp_change, d_weights);
        gsl_matrix_mul_elements(temp_change, old_d_weights);
        angle_delta = acos(matrix_sum(temp_change) / sqrt(matrix_norm(d_weights)*(matrix_norm(old_d_weights))));
        angle_delta *= (180.0 / M_PI);
      }

      gsl_matrix_memcpy(old_weights, weights);

      if (angle_delta > 60){
        lrate *= ANNEAL;
        gsl_matrix_memcpy(old_d_weights, d_weights);
      } else if (step==1) {
        gsl_matrix_memcpy(old_d_weights, d_weights);
      }

      if ((verbose && (step % 10)== 0) || change < W_STOP){
        printf("\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f",
          step, lrate, change, angle_delta);
      }

      step ++;
    }
  }

  matrix_mmul(weights, x_white, S);
  gsl_matrix_free(old_d_weights);
  gsl_matrix_free(old_weights);
  gsl_matrix_free(bias);
  gsl_matrix_free(d_weights);

}

void ica(gsl_matrix *A, gsl_matrix *S, gsl_matrix *X, int verbose){

  /* Checking the existance of the enviroment variable
  for controlling the number of threads used by openblas*/
  int success = setenv ("OPENBLAS_NUM_THREADS", "8", 1);
  if (success){
    printf("\nSetting OPENBLAS_NUM_THREADS to 8");
    printf("\nSet the enviroment variable to your number of cores");
  }
  else{

  }

  const size_t NCOMP = A->size2;
  const size_t NSUB = X->size1;
  const size_t NVOX = X->size2;
  gsl_matrix *weights = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *inv_weights = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *white_X = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *white   = gsl_matrix_alloc(NCOMP, NSUB);
  gsl_matrix *dewhite = gsl_matrix_alloc(NSUB, NCOMP);

  double start, end;
  double cpu_time_used;
  if (verbose) printf("\nPCA decomposition ...");
  start = omp_get_wtime();
  pca_whiten(X, NCOMP, white_X, white, dewhite, 1);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tPCA Time  %g, ", cpu_time_used);

  if (verbose) printf("Done.");
  if (verbose) printf("\nINFOMAX ...");
  if (verbose) printf("\nPCA decomposition ...");
  start = omp_get_wtime();
  infomax(white_X, weights, S, verbose);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tInfomax Time  %g, ", cpu_time_used);
  if (verbose) printf("Done");

  matrix_inv(weights, inv_weights);
  matrix_mmul(dewhite, inv_weights, A);

  gsl_matrix_free(weights);
  gsl_matrix_free(white_X);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);

}

/*
void infomax1(double *x_white){

}
*/
