#include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "util/util.h"
typedef int bool;
#define true 1
#define false 0

double EPS = 1e-18;
double MAX_W = 1e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;



void pca_whiten(gsl_matrix *x2d,  size_t const n_comp,
                gsl_matrix *x_white,
                gsl_matrix *white,
                gsl_matrix *dewhite,
                bool demean=true,
                bool verbose=true){
  // demean input matrix
  if (demean){
    matrix_demean(x2d);
  }
  gsl_matrix *cov = matrix_cov(x2d);
  // Set up eigen decomposition
  gsl_vector *eval; //eigen values
  gsl_matrix *evec; //eigen vector
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (cov->size1);
  gsl_eigen_symmv(cov, eval, evec, w);
  gsl_eigen_symm_free(w);
  // sort by eigen values
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
  // cut out the first n_comp components
  gsl_matrix_view reduced_evec = gsl_matrix_submatrix(evec, 0, 0, evec->size1, n_comp );

  gsl_matrix *white = gsl_matrix_alloc(x2d->size1, n_comp);
  gsl_blas_dgemm (CblasTrans, CblasNoTrans,
    1.0, eval, &reduced_evec.matrix, 0.0, white);

}
void w_update(double *unmixer, double *x_white, double *bias1,
              double *lrate1, double *error){

              }
void infomax1(double *x_white, bool verbose){

}
