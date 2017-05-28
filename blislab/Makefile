CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS -I.
LDLIBS = -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic

all:
	$(CC) $(CFLAGS) bl_sgemm_ref.c bl_sgemm_util.c my_sgemm.c bl_sgemm_asm_24x4.c test_bl_sgemm.c $(LDLIBS)
clean:
	rm -f a.out
