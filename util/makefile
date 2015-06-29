CC=gcc
LIBS=-lopenblas -llapack -lcunit -lgsl
CFLAGS=-Wall

util.o: util.c
	$(CC) -c util.c $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o *~ core
