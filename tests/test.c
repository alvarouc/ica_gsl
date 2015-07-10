// #include <CUnit/CUnit.h>
// #include <lapacke_utils.h>
#include <CUnit/Basic.h>
#include "../util/util.h"
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include "../ica.h"
#include <gsl/gsl_math.h>
#include <omp.h>

// Input matrix
size_t NROW = 500, NCOL = 100000;
size_t NSUB = 1000;
size_t NCOMP = 100;
size_t NVOX = 50000;
gsl_matrix *input, *true_A, *true_S, *true_X, *white_x, *white, *dewhite;
double start, end;
double cpu_time_used;
// check if memory was allocated

// unit testing for ICA
int init_suite_util(void){
  input = gsl_matrix_alloc(NROW, NCOL);
  if (NULL==input) return 1;
  gsl_matrix_set_all(input, 1.0);
  return 0;
}

int clean_suite_util(void){
  gsl_matrix_free(input);
  return 0;
}

int init_suite_ica(void){

  true_A = gsl_matrix_alloc(NSUB, NCOMP);
  true_S = gsl_matrix_alloc(NCOMP, NVOX);
  true_X = gsl_matrix_alloc(NSUB,NVOX);
  white_x = gsl_matrix_alloc(NCOMP, NVOX);
  white = gsl_matrix_alloc(NCOMP, NSUB);
  dewhite = gsl_matrix_alloc(NSUB,NCOMP);
  gsl_matrix *noise = gsl_matrix_alloc(NSUB, NVOX);
  random_matrix(noise, 0.5, gsl_ran_gaussian);
  // Random gaussian mixing matrix A
  random_matrix(true_A, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_S, 1.0, gsl_ran_logistic);
  // X = AS
  matrix_mmul(true_A, true_S, true_X);
  gsl_matrix_add(true_X, noise);
  gsl_matrix_free(noise);

  return 0;
}

int clean_suite_ica(void){
  gsl_matrix_free(true_A);
  gsl_matrix_free(true_S);
  gsl_matrix_free(true_X);
  gsl_matrix_free(white);
  gsl_matrix_free(white_x);
  gsl_matrix_free(dewhite);
  return 0;
}

void test_matrix_inv(void){

  double m[2][2] = {{10,0},{0,10}};
  double inv_m[2][2] = {{0.1,0},{0,0.1}};
  gsl_matrix_view x = gsl_matrix_view_array(&m[0][0],2,2);
  gsl_matrix_view xx = gsl_matrix_view_array(&inv_m[0][0],2,2);
  gsl_matrix *inv_x = gsl_matrix_alloc(2,2);
  matrix_inv(&x.matrix,inv_x);
  if (gsl_matrix_equal(inv_x, &xx.matrix))
    CU_PASS("Inverse is working");

}

void test_random_vector(void){

  gsl_vector *vec = gsl_vector_calloc(100000);
  random_vector(vec, 2.0, gsl_ran_gaussian);
  // print_vector_head(vec);

  double mean = gsl_stats_mean (vec->data, vec->stride, vec->size);
  double sd = gsl_stats_sd (vec->data, vec->stride, vec->size);
  // printf("* Mean %.2e, SD %.2f", mean, sd);
  CU_ASSERT(mean < 0.01);
  CU_ASSERT(abs(sd - 2) < 0.01);

  gsl_vector *vec2 = gsl_vector_calloc(100000);
  random_vector(vec2, 2.0, gsl_ran_gaussian);
  if (gsl_vector_equal(vec,vec2))
    CU_FAIL("random vectors are always the same");

  gsl_vector_free(vec);
  gsl_vector_free(vec2);

}

void test_random_matrix(void){
  gsl_matrix *vec = gsl_matrix_alloc(1000,1000);
  random_matrix(vec, 2.0, gsl_ran_gaussian);

  gsl_matrix *vec2 = gsl_matrix_alloc(1000,1000);
  random_matrix(vec2, 2.0, gsl_ran_gaussian);
  if (gsl_matrix_equal(vec,vec2))
    CU_FAIL("random matrices are always the same");

  gsl_matrix_free(vec);
  gsl_matrix_free(vec2);

}

void test_matrix_cross_corr(void){
  size_t NSUB = 10000;
  size_t NVAR = 100;
  gsl_matrix *A = gsl_matrix_alloc(NSUB, NVAR);
  gsl_matrix *A_T = gsl_matrix_alloc(NVAR, NSUB);
  gsl_matrix *test = gsl_matrix_alloc(NVAR,NVAR);
  gsl_matrix_set_identity(test);
  random_matrix(A, 1, gsl_ran_gaussian);
  gsl_matrix_transpose_memcpy(A_T, A);
  gsl_matrix *C = gsl_matrix_calloc(NVAR,NVAR);

  start = omp_get_wtime();
  matrix_cross_corr(C, A, A);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tBy col: Time  %g, ", cpu_time_used);

  gsl_matrix_sub(test, C);
  if ( matrix_norm(test)/NVAR/NVAR > 0.01 )
  { print_matrix_corner(C);
    CU_FAIL("correlation matrix is not close to identity!");
  }

  start = omp_get_wtime();
  matrix_cross_corr_row(C, A_T, A_T);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf(" By row : Time  %g, ", cpu_time_used);
  gsl_matrix_set_identity(test);
  gsl_matrix_sub(test, C);
  if ( matrix_norm(test)/NVAR/NVAR > 0.01 )
  { print_matrix_corner(C);
    CU_FAIL("correlation matrix is not close to identity!");
  }

  gsl_matrix_free(C);
  gsl_matrix_free(A);
  gsl_matrix_free(A_T);
  gsl_matrix_free(test);
}


void test_matrix_mean(void){
  // Test the util function matrix_mean

  // print_matrix_corner(input);
  gsl_matrix_set_all(input, 1);
  // Compute column mean
  gsl_vector *mean = gsl_vector_alloc(NCOL);

  start = omp_get_wtime();
  matrix_mean(mean, input);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  // Compare to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  gsl_vector_set_all(expected_mean, 1.0);
  CU_ASSERT(gsl_vector_equal(mean, expected_mean));
  gsl_vector_free(mean);
  // print_vector_head(mean);
}

void test_matrix_demean(void){
  // Test the util function matrix_demean
  start = omp_get_wtime();
  matrix_demean(input);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  gsl_vector *mean = gsl_vector_alloc(NCOL);
  matrix_mean(mean, input);

  // Compare mean to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  gsl_vector_set_all(expected_mean, 0.0);

  CU_ASSERT(gsl_vector_equal(mean, expected_mean));
  gsl_vector_free(mean);
  gsl_vector_free(expected_mean);
}

void test_matrix_cov(void){
  size_t NSUB = 100000;
  size_t NVAR = 100;
  gsl_matrix *A = gsl_matrix_alloc(NVAR, NSUB);
  gsl_matrix *cov = gsl_matrix_alloc(NVAR, NVAR);

  start = omp_get_wtime();
  matrix_cov(A, cov);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  // print_matrix_corner(cov);
  size_t i,j;
  size_t n_diferent = 0;
  for (i = 0; i < cov->size1; i++) {
    for (j = 0; j < cov->size2; j++) {
      if (gsl_matrix_get(cov,i,j)!=gsl_matrix_get(cov,j,i))
        n_diferent ++;
    }
  }
  CU_ASSERT_EQUAL(n_diferent, 0 );
}

void test_matrix_norm(void){

  gsl_matrix_set_all(input,1);

  start = omp_get_wtime();
  double norm = matrix_norm(input);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  // printf("\nMatrix Norm %g\n", norm);
  CU_ASSERT_EQUAL(norm, NROW*NCOL);

  gsl_matrix_set_all(input,2);
  norm = matrix_norm(input);
  // printf("\nMatrix Norm %g\n", norm);
  CU_ASSERT_EQUAL(norm, NROW*NCOL*4);

}

void test_matrix_sum(void){
  gsl_matrix_set_all(input, 1);

  start = omp_get_wtime();
  double sum = matrix_sum(input);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  CU_ASSERT_EQUAL(sum, NROW*NCOL);
}

void test_matrix_mmul(void){

  double m1[2][2] = {{1,1},{2,2}};
  double m2[2][2] = {{3,0},{1,1}};
  double m3[2][2] = {{4,1},{8,2}};

  gsl_matrix *mul = gsl_matrix_alloc(2, 2);
  gsl_matrix_view m1_view = gsl_matrix_view_array(&m1[0][0],2,2);
  gsl_matrix_view m2_view = gsl_matrix_view_array(&m2[0][0],2,2);
  gsl_matrix_view m3_view = gsl_matrix_view_array(&m3[0][0],2,2);

  matrix_mmul(&m1_view.matrix, &m2_view.matrix, mul);
  if (gsl_matrix_equal(mul, &m3_view.matrix))
    CU_PASS("matrix dot product works");

  gsl_matrix_free(mul);
}

void test_matrix_apply_all(void){

  gsl_matrix_set_all(input, 100);

  start = omp_get_wtime();
  matrix_apply_all(input, log10);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  gsl_matrix *test = gsl_matrix_alloc(input->size1, input->size2);
  gsl_matrix_set_all(test, 2.0);
  gsl_matrix_sub(test, input);
  if (matrix_norm(test)>1e-6)
    CU_FAIL("Not the expected oputput.");
  gsl_matrix_free(test);
}

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */

  // PCA(X)
  start = omp_get_wtime();
  pca_whiten(true_X, NCOMP, white_x, white, dewhite, 0);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\t\tTime  %g, ", cpu_time_used);
  // test if covariance of ouput is identity
  gsl_matrix *cov = gsl_matrix_alloc(NCOMP,NCOMP);
  matrix_cov(white_x, cov);
  gsl_matrix *expected_cov = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix_set_identity(expected_cov);
  gsl_matrix_sub(cov, expected_cov);
  CU_ASSERT(matrix_norm(cov)<1e-6);
  //test reconstruction error
  gsl_matrix *reconstructed_x = gsl_matrix_alloc(NSUB, NVOX);
  matrix_mmul(dewhite, white_x, reconstructed_x);

  gsl_matrix_sub(reconstructed_x, true_X);
  double reconstruction_error = matrix_norm(reconstructed_x)/NVOX/NSUB;
  if(reconstruction_error>1){
    printf("\nError : %g\n", reconstruction_error);
    CU_FAIL("PCA reconstruction error is too high");
  }

  gsl_matrix_free(reconstructed_x);
  gsl_matrix_free(cov);

}

void test_w_update(void){

  int success = setenv ("OPENBLAS_NUM_THREADS", "8", 1);
  if (success){
    printf("\nSetting OPENBLAS_NUM_THREADS to 8");
    printf("\nSet the enviroment variable to your number of cores");
  }


  gsl_matrix *weights = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *bias    = gsl_matrix_calloc(NCOMP,1);
  gsl_matrix *old_weights = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix *shuffled_x_white = gsl_matrix_alloc(NCOMP,NVOX);


  gsl_matrix_set_identity(weights);
  int error = 0;
  double lrate = 0.001;
  gsl_matrix_memcpy(old_weights, weights);

  start = omp_get_wtime();
  error = w_update(weights, white_x, bias, shuffled_x_white, lrate);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);

  CU_ASSERT_EQUAL(error, 0);
  if (gsl_matrix_equal(old_weights, weights))
    CU_FAIL("Weights have not been updated");

  lrate = 1000;
  error = w_update(weights, white_x, bias,shuffled_x_white, lrate);
  CU_ASSERT_EQUAL(error, 1);

  gsl_matrix_free(weights);
  gsl_matrix_free(old_weights);
  gsl_matrix_free(bias);
  gsl_matrix_free(shuffled_x_white);
}

void test_infomax(void){


  gsl_matrix *estimated_a = gsl_matrix_alloc(NSUB,  NCOMP);
  gsl_matrix *estimated_s = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *estimated_x = gsl_matrix_alloc(NSUB,  NVOX);
  gsl_matrix *cs          = gsl_matrix_alloc(NCOMP, NCOMP);

  // A,S <- ICA(X, NCOMP)
  start = omp_get_wtime();
  ica(estimated_a, estimated_s, true_X, 0);
  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));
  printf("\t\tTime  %g, ", cpu_time_used);
  // Match by correlation
  ica_match_gt(true_A, true_S, estimated_a, estimated_s);

  matrix_cross_corr_row(cs, true_S, estimated_s);
  printf("\nSource estimation accuracy");
  // print_matrix_corner(cs);
  matrix_apply_all(cs, absolute);
  gsl_vector_view diag = gsl_matrix_diagonal(cs);
  double avg = gsl_stats_mean(diag.vector.data, diag.vector.stride, diag.vector.size);
  printf("\n Average Accuracy: %.3f", avg);
  if(avg < 0.95){
    CU_FAIL("average source estiamtion accuracy too low.");
  }
  matrix_cross_corr(cs, true_A, estimated_a);
  printf("\nLoading estimation accuracy");
  // print_matrix_corner(cs);
  matrix_apply_all(cs, absolute);
  diag = gsl_matrix_diagonal(cs);
  avg = gsl_stats_mean(diag.vector.data, diag.vector.stride, diag.vector.size);
  printf("\n Average Accuracy: %.3f", avg);
  if(avg < 0.95){
    CU_FAIL("average source estiamtion accuracy too low.");
  }
  //Clean
  gsl_matrix_free(estimated_a);
  gsl_matrix_free(estimated_s);
  gsl_matrix_free(estimated_x);
  gsl_matrix_free(cs);
}


// functional testing for ICA
int main()
{
  CU_pSuite pSuite_util = NULL;
  CU_pSuite pSuite_ica = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite_util = CU_add_suite("Suite_util",
    init_suite_util, clean_suite_util);
   if (NULL == pSuite_util) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   pSuite_ica = CU_add_suite("Suite_ICA",
    init_suite_ica, clean_suite_ica);
   if (NULL == pSuite_ica) {
      CU_cleanup_registry();
      return CU_get_error();
   }


   /* add the tests to the suite */
   if (
(NULL == CU_add_test(pSuite_util,"test of matrix_cross_corr()",test_matrix_cross_corr)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_inv()",test_matrix_inv)) ||
(NULL == CU_add_test(pSuite_util,"test of random_vector()",test_random_vector)) ||
(NULL == CU_add_test(pSuite_util,"test of random_matrix()",test_random_matrix)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_mean()",test_matrix_mean)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_demean()",test_matrix_demean)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_cov()", test_matrix_cov)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_norm()", test_matrix_norm)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_sum()", test_matrix_sum)) ||
(NULL == CU_add_test(pSuite_util,"test of matrix_apply_all()", test_matrix_apply_all)) ||
(NULL == CU_add_test(pSuite_ica,"test whitening",test_pca_whiten)) ||
(NULL == CU_add_test(pSuite_ica,"test mixing matrix update",test_w_update))||
(NULL == CU_add_test(pSuite_ica,"test infomax",test_infomax))
      )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
}
