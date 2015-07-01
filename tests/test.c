#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// #include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "../util/util.h"
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include "../ica.h"
#include <gsl/gsl_randist.h>
// Input matrix
size_t NROW = 500, NCOL = 100000;
gsl_matrix *input;
// check if memory was allocated

// unit testing for ICA
int init_suite_util(void)
{
  input = gsl_matrix_alloc(NROW, NCOL);
  if (NULL==input) return 1;
  gsl_matrix_set_all(input, 1.0);
  return 0;
}

int clean_suite_util(void)
{
  gsl_matrix_free(input);
  return 0;
}

int init_suite_ica(void){
  input = gsl_matrix_alloc(NROW,NCOL);
  if (NULL==input) return 1;
  fill_matrix_random(input);
  matrix_demean(input);

  return 0;
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

void test_matrix_mean(void){
  // Test the util function matrix_mean

  // print_matrix_corner(input);

  // Compute column mean
  gsl_vector *mean = matrix_mean(input);
  CU_ASSERT_PTR_NOT_NULL(mean);
  // Compare to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  fill_vector_const(expected_mean, 1.0);
  CU_ASSERT(gsl_vector_equal(mean, expected_mean));

  // print_vector_head(mean);
}

void test_matrix_demean(void){
  // Test the util function matrix_demean

  matrix_demean(input);
  gsl_vector *mean = matrix_mean(input);

  // Compare mean to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  fill_vector_const(expected_mean, 0.0);

  CU_ASSERT(gsl_vector_equal(mean, expected_mean));
}

void test_matrix_cov(void){
  gsl_matrix *cov = gsl_matrix_alloc(input->size1, input->size1);
  matrix_cov(input, cov);
  // print_matrix_corner(cov);
  size_t i,j;
  size_t n_diferent = 0;
  for (i = 0; i < cov->size1; i++) {
    for (j = 0; j < cov->size2; j++) {
      if (gsl_matrix_get(cov,0,1)!=gsl_matrix_get(cov,1,0))
        n_diferent ++;
    }
  }
  CU_ASSERT_EQUAL(n_diferent, 0 );

}

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */
  gsl_matrix *x_white, *dewhite, *white;
  size_t NCOMP=10;
  white = gsl_matrix_alloc(NCOMP, input->size1);
  dewhite = gsl_matrix_alloc(input->size1, NCOMP);
  x_white = gsl_matrix_alloc(NCOMP, input->size2);

  pca_whiten(input, NCOMP, x_white, white, dewhite, 0);
  // test if covariance of ouput is identity
  gsl_matrix *cov = gsl_matrix_alloc(NCOMP,NCOMP);
  matrix_cov(x_white, cov);
  gsl_matrix *expected_cov = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix_set_identity(expected_cov);
  gsl_matrix_sub(cov, expected_cov);
  CU_ASSERT(matrix_norm(cov)<1e-6);

  gsl_matrix_free(cov);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);
  gsl_matrix_free(x_white);
}

void test_w_update(void){
  gsl_matrix *unmixer, *x_white, *bias;
  double rate = 1e-3;
  double *lrate;
  lrate = &rate;
  size_t NCOMP=10;

  gsl_matrix *dewhite, *white;
  white = gsl_matrix_alloc(NCOMP, input->size1);
  dewhite = gsl_matrix_alloc(input->size1, NCOMP);
  x_white = gsl_matrix_alloc(NCOMP, input->size2);

  pca_whiten(input, NCOMP, x_white, white, dewhite, 0);

  unmixer = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix_set_identity(unmixer);
  bias = gsl_matrix_calloc(NCOMP,1);
  int error = 0;
  error = w_update(unmixer, x_white, bias, lrate);
  CU_ASSERT_EQUAL(error, 0);

  *lrate = 1000;
  error = w_update(unmixer, x_white, bias, lrate);
  CU_ASSERT_EQUAL(error, 1);


  gsl_matrix_free(x_white);
  gsl_matrix_free(unmixer);
  gsl_matrix_free(white);
  gsl_matrix_free(bias);
  gsl_matrix_free(dewhite);
}

void test_infomax(void){
  size_t NCOMP = 3;
  size_t NVOX = 10000;
  gsl_matrix *true_A = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *true_S = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_X = gsl_matrix_alloc(NCOMP,NVOX);
  gsl_matrix *estimated_A = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *estimated_S = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *estimated_X = gsl_matrix_alloc(NCOMP,NVOX);

  // Random gaussian mixing matrix A
  random_matrix(true_A, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_S, 1.0, gsl_ran_logistic);
  // X = AS
  matrix_mmul(true_A, true_S, true_X);
  infomax(true_X, estimated_A, estimated_S);

  matrix_mmul(estimated_A, estimated_S, estimated_X);

  gsl_matrix_sub(true_X, estimated_X);
  if (matrix_norm(true_X)> 1.0e-6)
    CU_FAIL(printf("Matrix reconstruction error is %.2f",
      matrix_norm(true_X)));

  print_matrix_corner(true_A);

  gsl_matrix_free(true_A);
  gsl_matrix_free(true_S);
  gsl_matrix_free(true_X);
  gsl_matrix_free(estimated_A);
  gsl_matrix_free(estimated_S);
  gsl_matrix_free(estimated_X);

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
    init_suite_ica, clean_suite_util);
   if (NULL == pSuite_ica) {
      CU_cleanup_registry();
      return CU_get_error();
   }


   /* add the tests to the suite */
   if (
(NULL == CU_add_test(pSuite_util,
  "test of matrix_mean()",
  test_matrix_mean)) ||
(NULL == CU_add_test(pSuite_util,
  "test of matrix_demean()",
  test_matrix_demean)) ||
(NULL == CU_add_test(pSuite_util,
    "test of random_vector()",
    test_random_vector)) ||
(NULL == CU_add_test(pSuite_util,
  "test of matrix_cov()",
  test_matrix_cov)) ||
(NULL == CU_add_test(pSuite_ica,
  "test whitening",
  test_pca_whiten)) ||
(NULL == CU_add_test(pSuite_ica,
    "test mixing matrix update",
    test_w_update)) ||
(NULL == CU_add_test(pSuite_ica,
    "test infomax",
    test_infomax))
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
