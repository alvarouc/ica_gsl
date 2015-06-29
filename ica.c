#include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
typedef int bool;
#define true 1
#define false 0

double EPS = 1e-18;
double MAX_W = 1e8;
double ANNEAL = 0.9;
double MIN_LRATE = 1e-6;
double W_STOP = 1e-6;

void pca_whiten(double *x2d,  int n_comp,
                double *x_white, double *white,
                double *dewhite, double *u, double *s,
                double *v, bool verbose);
void w_update(double *unmixer, double *x_white, double *bias1,
              double *lrate1, double *error);
void infomax1(double *x_white, bool verbose);


int main(){
        return 0;

}
