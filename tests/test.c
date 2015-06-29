// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// #include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
// #include "ica.h"
#include "../util/util.h"
#include <gsl/gsl_eigen.h>

// Input matrix
size_t NROW = 1000, NCOL = 10000;
gsl_matrix *input;
// check if memory was allocated

// unit testing for ICA
int init_suite1(void)
{
  input = gsl_matrix_alloc(NROW, NCOL);
  if (NULL==input) return 1;

  return 0;
}

int clean_suite1(void)
{
  gsl_matrix_free(input);
  return 0;
}

void test_matrix_mean(void){
  // Test the util function matrix_mean
  fill_matrix_const(input, 1.0);
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
  fill_matrix_const(input, 1.0);

  matrix_demean(input);
  gsl_vector *mean = matrix_mean(input);

  // Compare mean to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  fill_vector_const(expected_mean, 0.0);

  CU_ASSERT(gsl_vector_equal(mean, expected_mean));
}

void test_matrix_cov(void){
  fill_matrix_random(input);
  matrix_demean(input);

  gsl_matrix *cov = matrix_cov(input);
  CU_ASSERT_PTR_NOT_NULL(cov);
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

void test_eigen_decomp(void){
  fill_matrix_random(input);
  // printf("Raw Input Matrix\n");
  // print_matrix_corner(input);

  matrix_demean(input);

  // printf("demeaned Matrix\n");
  // print_matrix_corner(input);
  gsl_matrix *cov = matrix_cov(input);
  // Set up eigen decomposition
  gsl_vector *eval = gsl_vector_alloc(cov->size1); //eigen values
  gsl_matrix *evec = gsl_matrix_alloc(cov->size1, cov->size2); //eigen vector
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (cov->size1);
  CU_ASSERT_PTR_NOT_NULL(w);
  gsl_eigen_symmv(cov, eval, evec, w);
  gsl_eigen_symmv_free(w);

  CU_ASSERT_PTR_NOT_NULL(eval);
  CU_ASSERT_PTR_NOT_NULL(evec);

  CU_FAIL("FInish the eigen test");
}

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */
  fill_matrix_random(input);
  print_matrix_corner(input);

  matrix_demean(input);
  // gsl_matrix *white, *dewhite, *x_white;
  // pca_whiten(input, 10, x_white, white, dewhite, 0);


  // Clean up
  CU_FAIL("Complete the test!");
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
   pSuite_util = CU_add_suite("Suite_util", init_suite1, clean_suite1);
   if (NULL == pSuite_util) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   pSuite_ica = CU_add_suite("Suite_ICA", init_suite1, clean_suite1);
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
  "test of matrix_cov()",
  test_matrix_cov)) ||
(NULL == CU_add_test(pSuite_ica,
  "test eigen decomposition",
  test_eigen_decomp))
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
