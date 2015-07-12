# CFLAGS=-L/opt/OpenBLAS/lib -I /opt/OpenBLAS/include
# -lopenblas -llapack -lgsl -lcunit
OPENBLAS_DIR=/home/aulloa/src/openblas-op
CC=gcc
LIBS_TEST=-lcunit
INC = -I$(OPENBLAS_DIR)/include
LIBS_UTIL=-lgsl -lm -fopenmp \
-L$(OPENBLAS_DIR)/lib -lopenblas 
CFLAGS= -Wall -Ofast
OBJ = util.o test.o ica.o

test: $(OBJ)
	$(CC) $^ -o test $(CFLAGS) $(LIBS_TEST) $(LIBS_UTIL)

util.o: ../util/util.c ../util/util.h
	$(CC) -c $(INC) ../util/util.c $(CFLAGS) $(LIBS_UTIL)

ica.o: ../ica.c ../ica.h util.o
	$(CC) -c $(INC) ../ica.c $(CFLAGS) $(LIBS_UTIL)

test.o: test.c util.o ica.o
	$(CC) -c $(INC) test.c $(CFLAGS) $(LIBS_UTIL) $(LIBS_TEST)



.PHONY: clean

clean:
	rm -f *.o *~ core
