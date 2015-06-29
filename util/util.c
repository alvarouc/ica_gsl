#include "utils.h"

void fill_matrix_random(gsl_matrix *input){

  NROW = input->size1;
  NCOL = input->size2;

  for (i = 0; i < NROW; i++)
    for (j = 0; j < NCOL; j++){
                  gsl_matrix_set(input, i, j, (double)(rand()));
          }

}

void print_matrix_corner(gsl_matrix *input){

  int i,j;

  printf ("Matrix size: %dx%d"\n, input->size1, input->size2)

  NROW = input->size1 < 3 ? input->size1 : 3;
  NCOL = input->size2 < 3 ? input->size2 : 3;

  for (i = 0; i < NROW; i++){
    for (size_t j = 0;j < NCOL; j++) {
      printf("%g ", gsl_matrix_get(m,i,j))
    }
    printf("\n")
  }

}
