CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -I.
LDLIBS = -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic

all:
	$(CC) $(CFLAGS) test.cpp $(LDLIBS)
	./run.sh
	@make --no-print-directory clean
clean:
	@rm -f a.out
