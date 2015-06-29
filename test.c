#include <lapacke.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <CUnit/CUnit.h>
#include "ica.h"
#include "utils.h"

// unit testing for ICA
int init_suite1(void)
{
   return 0
}
int clean_suite1(void)
{
   return 0
}

void test_pca_whiten(void)  {
  /*
  Test if pca_whiten function works as expected
  */
  int i,j;

  // Input matrix
  size_t NROW = 100, NCOL = 1000;
  gsl_matrix *input = gsl_matrix_alloc(NROW, NCOL);
  CU_ASSERT_PTR_NOT_NULL(input)

  CU_PASS()
}



  //output allocation
  output = gsl_matrix_alloc(m,n);
  // matrix multiplication
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,
                 1.0, input, input,
                 0.0, output);


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
   if ((NULL == CU_add_test(pSuite, "test of whitening", testFPRINTF)) )
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
