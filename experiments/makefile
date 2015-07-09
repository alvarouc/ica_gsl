# CFLAGS=-L/opt/OpenBLAS/lib -I /opt/OpenBLAS/include
# -lopenblas -llapack -lgsl -lcunit
CC=gcc
LIBS=-lgsl -lopenblas -lm -fopenmp -llapacke
# LIBS=-lgsl -lcblas -lm -fopenmp -llapacke
CFLAGS= -Wall -O3
OBJ = simple.o ica.o util.o

simple: $(OBJ)
	$(CC) $^ -o simple $(CFLAGS) $(LIBS)

util.o: ../util/util.c ../util/util.h
	$(CC) -c ../util/util.c $(CFLAGS) $(LIBS)

ica.o: ../ica.c ../ica.h
	$(CC) -c ../ica.c $(CFLAGS) $(LIBS)

simple.o: simple.c
	$(CC) -c simple.c $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o *~ core
