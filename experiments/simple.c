#include "../ica.h"
// #include "../util/util.h"
#include <time.h>
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
  clock_t start, end;
  double cpu_time_used;

  start = clock();
  // A,S <- ICA(X, NCOMP)
  ica(estimated_a, estimated_s, true_x, verbose);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Time used : %g", cpu_time_used);
  // Match by correlation
  ica_match_gt(true_a, true_s, estimated_a, estimated_s);

  matrix_cross_corr_row(cs, true_s, estimated_s);
  printf("\nSource estimation accuracy");
  print_matrix_corner(cs);
  matrix_apply_all(cs, absolute);
  gsl_vector_view diag = gsl_matrix_diagonal(cs);
  double avg = gsl_stats_mean(diag.vector.data, diag.vector.stride, diag.vector.size);
  printf("\n Average Accuracy: %.3f", avg);

  matrix_cross_corr(cs, true_a, estimated_a);
  printf("\nLoading estimation accuracy");
  print_matrix_corner(cs);
  matrix_apply_all(cs, absolute);
  diag = gsl_matrix_diagonal(cs);
  avg = gsl_stats_mean(diag.vector.data, diag.vector.stride, diag.vector.size);
  printf("\n Average Accuracy: %.3f", avg);

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
