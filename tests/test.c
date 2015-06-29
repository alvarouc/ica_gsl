// #include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// #include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
// #include "ica.h"
#include "../util/util.h"

// Input matrix
size_t NROW = 100, NCOL = 1000;
gsl_matrix *input;
// check if memory was allocated

// unit testing for ICA
int init_suite1(void)
{
  input = gsl_matrix_alloc(NROW, NCOL);
  CU_ASSERT_PTR_NOT_NULL(input);
  fill_matrix_random(input);
  return 0;
}

int clean_suite1(void)
{
  gsl_matrix_free(input)
  return 0;
}

void test_matrix_mean(void){
  // Test the util function matrix_mean
  gsl_vector *mean = matrix_mean(input);
  CU_ASSERT_PTR_NOT_NULL(mean);
  

}

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */

  print_matrix_corner(input);
  // COV(input)

  // Clean up
  CU_FAIL("Complete the test!");
}

// functional testing for ICA
int main()
{
   CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Suite_1", init_suite1, clean_suite1);
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of whitening", test_pca_whiten)) )
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
