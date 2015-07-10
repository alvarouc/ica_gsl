// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <omp.h>

#include "util/util.h"
void pca_whiten(gsl_matrix *input,  size_t const NCOMP,
                gsl_matrix *x_white,
                gsl_matrix *white,
                gsl_matrix *dewhite,
                int demean);
int w_update( gsl_matrix *weights,
              gsl_matrix *x_white,
              gsl_matrix *bias,
              gsl_matrix *shuffled_x_white, //work space for shuffled x_white
              gsl_permutation *p, // random permutation
              gsl_rng *r, // random stream from gsl
              double lrate);

void infomax(gsl_matrix *x_white, gsl_matrix *A, gsl_matrix *S, int verbose);
void ica(gsl_matrix *mixing, gsl_matrix *sources, gsl_matrix *data, int verbose);
