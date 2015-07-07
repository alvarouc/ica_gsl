#include "../ica.h"
// #include "../util/util.h"
#include <time.h>
#include <opm.h>

// #include <gsl/gsl_matrix.h>
double experiment(size_t, size_t, size_t, int);

double experiment(size_t NSUB, size_t NCOMP, size_t NVOX, int verbose){

  gsl_matrix *estimated_a = gsl_matrix_alloc(NSUB,  NCOMP);
  gsl_matrix *estimated_s = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *estimated_x = gsl_matrix_alloc(NSUB,  NVOX);
  gsl_matrix *true_a      = gsl_matrix_alloc(NSUB,  NCOMP);
  gsl_matrix *true_s      = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_x      = gsl_matrix_alloc(NSUB,  NVOX);
  gsl_matrix *cs          = gsl_matrix_alloc(NCOMP, NCOMP);

  // Random gaussian mixing matrix A
  random_matrix(true_a, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_s, 1.0, gsl_ran_logistic);
  // matrix_apply_all(true_s, gsl_pow_3);
  // X = AS
  matrix_mmul(true_a, true_s, true_x);
  double start, end;
  double cpu_time_used;

  start = omp_get_wtime();
  // A,S <- ICA(X, NCOMP)
  ica(estimated_a, estimated_s, true_x, verbose);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\nTime used : %g\n", cpu_time_used);

  //Clean
  gsl_matrix_free(true_a);
  gsl_matrix_free(true_s);
  gsl_matrix_free(true_x);
  gsl_matrix_free(estimated_a);
  gsl_matrix_free(estimated_s);
  gsl_matrix_free(estimated_x);
  gsl_matrix_free(cs);

  return (cpu_time_used);
}

int main(int argc, char const *argv[]) {
  size_t NSUB = 400;
  size_t NCOMP = 10;
  size_t NVOX = 10000;
  double cputime=0;

  size_t i, repetitions= 100;
  for (i = 0; i < repetitions; i++) {
    cputime += experiment(NSUB, NCOMP, NVOX, 0);
  }

  printf("\nCPU time used : %g", cputime/(double)repetitions);

  return 0;
}