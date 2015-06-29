# CFLAGS=-L/opt/OpenBLAS/lib -I /opt/OpenBLAS/include
# -lopenblas -llapack -lgsl -lcunit
CC=gcc
LIBS_TEST=-lopenblas -llapack -lcunit -lgsl
LIBS_UTIL=-lopenblas
CFLAGS= -Wall
# DEPS = ../util/utils.h
OBJ = util.o test.o

test: $(OBJ)
	$(CC) $^ -o test $(CFLAGS) $(LIBS_TEST)

util.o: ../util/util.c ../util/util.h
	$(CC) -c ../util/util.c $(CFLAGS) $(LIBS_UTIL)

test.o: test.c
	$(CC) -c test.c $(CFLAGS) $(LIBS_TEST)



.PHONY: clean

clean:
	rm -f *.o *~ core
