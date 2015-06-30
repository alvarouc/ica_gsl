// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
void pca_whiten(gsl_matrix *input,  size_t const NCOMP,
                gsl_matrix *x_white,
                gsl_matrix *white,
                gsl_matrix *dewhite,
                int demean);
/*void w_update(double *unmixer, double *x_white, double *bias1,
              double *lrate1, double *error);
void infomax1(double *x_white, bool verbose);
*/
