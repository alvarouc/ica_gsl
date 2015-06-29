#include "util.h"

void fill_matrix_random(gsl_matrix *input){
// Fill a GSL matrix with random numbers
  size_t NROW = input->size1;
  size_t NCOL = input->size2;

  int i,j;

  for (i = 0; i < NROW; i++)
    for (j = 0; j < NCOL; j++){
                  gsl_matrix_set(input, i, j, (double)(rand()));
          }

}

void print_matrix_corner(gsl_matrix *input){

  int i,j;

  printf ("\n\nMatrix size: %zux%zu\n", input->size1, input->size2);

  size_t NROW = input->size1 < 3 ? input->size1 : 3;
  size_t NCOL = input->size2 < 3 ? input->size2 : 3;

  for (i = 0; i < NROW; i++){
    for (j = 0;j < NCOL; j++) {
      printf("%g ", gsl_matrix_get(input,i,j));
    }
    printf("\n");
  }

}
