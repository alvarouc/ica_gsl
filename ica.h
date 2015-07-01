// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
void pca_whiten(gsl_matrix *input,  size_t const NCOMP,
                gsl_matrix *x_white,
                gsl_matrix *white,
                gsl_matrix *dewhite,
                int demean);
int w_update(gsl_matrix *unmixer, gsl_matrix *x_white,
              gsl_matrix *bias1, double *lrate1);

void infomax(gsl_matrix *x_white, gsl_matrix *A, gsl_matrix *S);
