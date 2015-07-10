# CFLAGS=-L/opt/OpenBLAS/lib -I /opt/OpenBLAS/include
# -lopenblas -llapack -lgsl -lcunit
CC=gcc
INC = -I/home/aulloa/src/openblas-op/include
LIBS=-lgsl -lm -fopenmp -llapacke -L/home/aulloa/src/openblas-op/lib -lopenblas
CFLAGS= -Wall -Ofast
OBJ = simple.o ica.o util.o

simple: $(OBJ)
	$(CC) $^ -o simple $(CFLAGS) $(LIBS)

util.o: ../util/util.c ../util/util.h
	$(CC) $(INC) -c ../util/util.c $(CFLAGS) $(LIBS)

ica.o: ../ica.c ../ica.h util.o
	$(CC) $(INC) -c ../ica.c $(CFLAGS) $(LIBS)

simple.o: simple.c util.o ica.o
	$(CC) $(INC) -c simple.c $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o *~ core
