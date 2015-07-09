# ica_gsl
Independent component component analysis

This is an ICA implementation using the infomax algorithm with whitening and PCA reduction.

Current Dependencies
 - GNU GSL
 - OpenBLAS (Compile with "$make USE_OPENMP=1")
 - OpenMP

The experiments folder contains a use example in the experiments/simple.c file.

## Configuration
In the makefile please change the include and library directories to your correspondent folder. 
