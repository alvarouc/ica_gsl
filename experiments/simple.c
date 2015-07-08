#include "../ica.h"
#include <time.h>
#include <omp.h>

double experiment(size_t NSUB, size_t NCOMP, size_t NVOX, int verbose){

  gsl_matrix *estimated_a = gsl_matrix_alloc(NSUB,  NCOMP);
  gsl_matrix *estimated_s = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *estimated_x = gsl_matrix_alloc(NSUB,  NVOX);
  gsl_matrix *true_a      = gsl_matrix_alloc(NSUB,  NCOMP);
  gsl_matrix *true_s      = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_x      = gsl_matrix_alloc(NSUB,  NVOX);
  gsl_matrix *cs          = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *noise       = gsl_matrix_alloc(NSUB,  NVOX);

  // Random gaussian mixing matrix A
  random_matrix(true_a, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_s, 1.0, gsl_ran_logistic);
  // Random gaussian noise
  random_matrix(noise, 1, gsl_ran_gaussian);
  // matrix_apply_all(true_s, gsl_pow_3);
  // X = AS
  matrix_mmul(true_a, true_s, true_x);
  // add noise
  gsl_matrix_add(true_x, noise);

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
  size_t NSUB = 1000;
  size_t NCOMP = 100;
  size_t NVOX = 50000;
  double cputime=0;

  size_t i, repetitions= 1;
  for (i = 0; i < repetitions; i++) {
    cputime += experiment(NSUB, NCOMP, NVOX, 0);
  }

  printf("\nCPU time used : %g", cputime/(double)repetitions);

  return 0;
}
