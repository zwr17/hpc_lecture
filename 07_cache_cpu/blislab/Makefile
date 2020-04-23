CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -I.
LDLIBS = -lpthread -lm -qopenmp -mkl=parallel -Werror -Wall -pedantic

test:
	$(CC) $(CFLAGS) test.cpp $(LDLIBS)

clean:
	@rm -f a.out
