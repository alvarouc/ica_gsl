// #include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "../util/util.h"
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include "../ica.h"
#include <gsl/gsl_math.h>

// Input matrix
size_t NROW = 500, NCOL = 100000;
gsl_matrix *input;
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
  input = gsl_matrix_alloc(NROW,NCOL);
  if (NULL==input) return 1;
  random_matrix(input, 1, gsl_ran_logistic);
  matrix_demean(input);

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

void test_matrix_corr(void){
  size_t NSUB = 1000;
  size_t NVAR = 10;
  gsl_matrix *A = gsl_matrix_alloc(NSUB, NVAR);
  gsl_matrix *test = gsl_matrix_alloc(NVAR,NVAR);
  gsl_matrix_set_identity(test);
  random_matrix(A, 1, gsl_ran_gaussian);

  gsl_matrix *C = gsl_matrix_calloc(NVAR,NVAR);

  matrix_cross_corr(C, A, A);

  gsl_matrix_sub(test, C);
  if ( matrix_norm(test)/NVAR/NVAR > 0.01 )
    CU_FAIL("correlation matrix is not close to identity!");

  gsl_matrix_free(C);
}


void test_matrix_mean(void){
  // Test the util function matrix_mean

  // print_matrix_corner(input);

  // Compute column mean
  gsl_vector *mean = matrix_mean(input);
  CU_ASSERT_PTR_NOT_NULL(mean);
  // Compare to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  gsl_vector_set_all(expected_mean, 1.0);
  CU_ASSERT(gsl_vector_equal(mean, expected_mean));

  // print_vector_head(mean);
}

void test_matrix_demean(void){
  // Test the util function matrix_demean

  matrix_demean(input);
  gsl_vector *mean = matrix_mean(input);

  // Compare mean to the expected mean
  gsl_vector *expected_mean = gsl_vector_alloc(NCOL);
  gsl_vector_set_all(expected_mean, 0.0);

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
      if (gsl_matrix_get(cov,i,j)!=gsl_matrix_get(cov,j,i))
        n_diferent ++;
    }
  }
  CU_ASSERT_EQUAL(n_diferent, 0 );
}

void test_matrix_norm(void){

  gsl_matrix_set_all(input,1);
  double norm = matrix_norm(input);
  // printf("\nMatrix Norm %g\n", norm);
  CU_ASSERT_EQUAL(norm, NROW*NCOL);

  gsl_matrix_set_all(input,2);
  norm = matrix_norm(input);
  // printf("\nMatrix Norm %g\n", norm);
  CU_ASSERT_EQUAL(norm, NROW*NCOL*4);

}

void test_matrix_sum(void){
  gsl_matrix_set_all(input, 1);
  double sum = matrix_sum(input);
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
  matrix_apply_all(input, log10);

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
  size_t NSUB = 200;
  size_t NCOMP = 3;
  size_t NVOX = 10000;
  gsl_matrix *true_A = gsl_matrix_alloc(NSUB, NCOMP);
  gsl_matrix *true_S = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_X = gsl_matrix_alloc(NSUB,NVOX);
  gsl_matrix *white_x = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *white = gsl_matrix_alloc(NCOMP, NSUB);
  gsl_matrix *dewhite = gsl_matrix_alloc(NSUB,NCOMP);

  // Random gaussian mixing matrix A
  random_matrix(true_A, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_S, 1.0, gsl_ran_logistic);
  // X = AS
  matrix_mmul(true_A, true_S, true_X);
  // PCA(X)
  pca_whiten(true_X, NCOMP, white_x, white, dewhite, 0);
  // test if covariance of ouput is identity
  gsl_matrix *cov = gsl_matrix_alloc(NCOMP,NCOMP);
  matrix_cov(white_x, cov);
  gsl_matrix *expected_cov = gsl_matrix_alloc(NCOMP,NCOMP);
  gsl_matrix_set_identity(expected_cov);
  gsl_matrix_sub(cov, expected_cov);
  CU_ASSERT(matrix_norm(cov)<1e-6);
  //test reconstruction error
  gsl_matrix *reconstructed_x = gsl_matrix_alloc(true_X->size1, true_X->size2);
  matrix_mmul(dewhite, white_x, reconstructed_x);

  gsl_matrix_sub(reconstructed_x, true_X);
  double reconstruction_error = matrix_norm(reconstructed_x);
  if(reconstruction_error>1e-6){
    CU_FAIL("PCA reconstruction error is too high");
    printf("\nError : %g\n", reconstruction_error);
  }

  gsl_matrix_free(reconstructed_x);
  gsl_matrix_free(cov);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);
  gsl_matrix_free(white_x);
  gsl_matrix_free(true_X);
  gsl_matrix_free(true_S);
  gsl_matrix_free(true_A);

}

void test_w_update(void){

  size_t NSUB = 200;
  size_t NCOMP = 3;
  size_t NVOX = 10000;
  gsl_matrix *true_A  = gsl_matrix_alloc(NSUB, NCOMP);
  gsl_matrix *true_S  = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_X  = gsl_matrix_alloc(NSUB,NVOX);
  gsl_matrix *white_x = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *white   = gsl_matrix_alloc(NCOMP, NSUB);
  gsl_matrix *dewhite = gsl_matrix_alloc(NSUB,NCOMP);
  gsl_matrix *weights = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *bias    = gsl_matrix_calloc(NCOMP,1);
  gsl_matrix *old_weights = gsl_matrix_alloc(NCOMP,NCOMP);
  // Random gaussian mixing matrix A
  random_matrix(true_A, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_S, 1.0, gsl_ran_logistic);
  // X = AS
  matrix_mmul(true_A, true_S, true_X);
  // PCA(X)

  gsl_matrix *old_true_X = gsl_matrix_alloc(NSUB,NVOX);
  gsl_matrix_memcpy(old_true_X, true_X);
  pca_whiten(true_X, NCOMP, white_x, white, dewhite, 0);

  // Check if white_x was modified
  if(gsl_matrix_equal(old_true_X, true_X)==0)
    CU_FAIL("PCA_WHITEN modified its input!");
  gsl_matrix_free(old_true_X);

  gsl_matrix_set_identity(weights);
  int error = 0;
  double lrate = 0.001;
  gsl_matrix_memcpy(old_weights, weights);
  error = w_update(weights, white_x, bias, lrate);
  CU_ASSERT_EQUAL(error, 0);
  if (gsl_matrix_equal(old_weights, weights))
    CU_FAIL("Weights have not been updated");

  lrate = 1000;
  error = w_update(weights, white_x, bias, lrate);
  CU_ASSERT_EQUAL(error, 1);

  gsl_matrix_free(true_A);
  gsl_matrix_free(true_S);
  gsl_matrix_free(true_X);
  gsl_matrix_free(white_x);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);
  gsl_matrix_free(weights);
  gsl_matrix_free(bias);
}

void test_infomax(void){

  size_t NSUB = 200;
  size_t NCOMP = 3;
  size_t NVOX = 10000;

  gsl_matrix *estimated_a = gsl_matrix_alloc(NCOMP, NCOMP);
  gsl_matrix *es_dewh_a   = gsl_matrix_alloc(NSUB, NCOMP);
  gsl_matrix *estimated_s = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *estimated_x = gsl_matrix_alloc(NCOMP,NVOX);
  gsl_matrix *true_a      = gsl_matrix_alloc(NSUB, NCOMP);
  gsl_matrix *true_s      = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *true_x      = gsl_matrix_alloc(NSUB,NVOX);
  gsl_matrix *white_x     = gsl_matrix_alloc(NCOMP, NVOX);
  gsl_matrix *white       = gsl_matrix_alloc(NCOMP, NSUB);
  gsl_matrix *dewhite     = gsl_matrix_alloc(NSUB,NCOMP);

  // Random gaussian mixing matrix A
  random_matrix(true_a, 1.0, gsl_ran_gaussian);
  // Random logistic mixing matrix S
  random_matrix(true_s, 1.0, gsl_ran_logistic);
  matrix_apply_all(true_s, gsl_pow_3);

  // X = AS
  matrix_mmul(true_a, true_s, true_x);
  // PCA(X)
  pca_whiten(true_x, NCOMP, white_x, white, dewhite, 1);
  // Run infomax on whitened data
  infomax(white_x, estimated_a, estimated_s);
  // dewhite estimated A matrix
  matrix_mmul(dewhite, estimated_a, es_dewh_a);
  // Test accuracy of source estimation
  gsl_matrix *cs = gsl_matrix_alloc(NCOMP, NCOMP);
  matrix_cross_corr_row(cs, estimated_s, true_s);
  printf("\nSource Accuracy");
  print_matrix_corner(cs);
  // Test accuracy of loading estimation
  matrix_cross_corr(cs, es_dewh_a, true_a);
  printf("\nLoading Accuracy");
  print_matrix_corner(cs);

  //Clean
  gsl_matrix_free(true_a);
  gsl_matrix_free(true_s);
  gsl_matrix_free(true_x);
  gsl_matrix_free(estimated_a);
  gsl_matrix_free(estimated_s);
  gsl_matrix_free(estimated_x);
  gsl_matrix_free(white);
  gsl_matrix_free(dewhite);
  gsl_matrix_free(white_x);
  gsl_matrix_free(cs);
  gsl_matrix_free(es_dewh_a);
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
(NULL == CU_add_test(pSuite_util,"test of matrix_cross_corr()",test_matrix_corr)) ||
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
