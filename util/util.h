
#ifndef UTILS_H_   /* Include guard */
#define UTILS_H_
// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics.h>

// operations
void matrix_cross_corr(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B);
void matrix_cross_corr_row(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B);
void matrix_inv(gsl_matrix *input, gsl_matrix *output);
void random_vector(gsl_vector *vec,  double parameter, double (* func)(const gsl_rng *, double ));
void random_matrix(gsl_matrix *vec, double parameter,double (* func)(const gsl_rng *, double ));
void matrix_mean(gsl_vector *mean, gsl_matrix *input);
void matrix_demean(gsl_matrix *input);
void matrix_cov(const gsl_matrix *input, gsl_matrix *cov);
double matrix_norm(gsl_matrix *input);
double matrix_sum(gsl_matrix *input);
void matrix_mmul(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C);
void matrix_apply_all(gsl_matrix *input, double (*fun)(double));
void ica_match_gt(gsl_matrix *true_a, gsl_matrix *true_s,
  gsl_matrix *esti_a, gsl_matrix *esti_s);
double absolute(double value);
// matrix print
void print_matrix_corner(gsl_matrix *input);
void print_vector_head(gsl_vector *input);

// void matrix_pinv(gsl_matrix *input, gsl_matrix *output)

#endif // FOO_H_
