# CFLAGS=-L/opt/OpenBLAS/lib -I /opt/OpenBLAS/include
# -lopenblas -llapack -lgsl -lcunit
CC=gcc
LIBS_TEST=-lcunit
# LIBS_UTIL=-lgsl -lcblas -lm -fopenmp -llapacke
# LIBS_UTIL=-lgsl -lopenblas -lm -fopenmp -llapacke
INC = -I/home/aulloa/src/openblas-op/include
LIBS_UTIL=-lgsl -lm -fopenmp -L/home/aulloa/src/openblas-op/lib -lopenblas -llapacke
CFLAGS= -Wall
# DEPS = ../util/utils.h
OBJ = util.o test.o ica.o

test: $(OBJ)
	$(CC) $^ -o test $(CFLAGS) $(LIBS_TEST) $(LIBS_UTIL)

util.o: ../util/util.c ../util/util.h
	$(CC) -c $(INC) ../util/util.c $(CFLAGS) $(LIBS_UTIL)

ica.o: ../ica.c ../ica.h
	$(CC) -c $(INC) ../ica.c $(CFLAGS) $(LIBS_UTIL)

test.o: test.c
	$(CC) -c $(INC) test.c $(CFLAGS) $(LIBS_UTIL) $(LIBS_TEST)



.PHONY: clean

clean:
	rm -f *.o *~ core
