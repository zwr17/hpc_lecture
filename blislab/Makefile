CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS -I.
LDLIBS = -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic

all:
	$(CC) $(CFLAGS) test_bl_sgemm.c $(LDLIBS)
clean:
	rm -f a.out
