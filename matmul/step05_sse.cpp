#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>
#include "pcounter.h"
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  float ** A = new float * [N];
  float ** B = new float * [N];
  float ** C = new float * [N];
  for (int i=0; i<N; i++) {
    A[i] = new float [N];
    B[i] = new float [N];
    C[i] = new float [N];
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  startPAPI();
  double tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      __m128 Aik = _mm_set1_ps(A[i][k]);
      for (int j=0; j<N; j+=4) {
        __m128 Cij = _mm_load_ps(&C[i][j]);
        __m128 Bkj = _mm_load_ps(&B[k][j]);
        Bkj = _mm_mul_ps(Aik, Bkj);
        Cij = _mm_add_ps(Bkj, Cij);
        _mm_store_ps(&C[i][j], Cij);
      }
    }
  }
  double toc = get_time();
  stopPAPI();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      __m128 Aik = -_mm_set1_ps(A[i][k]);
      for (int j=0; j<N; j+=4) {
        __m128 Cij = _mm_load_ps(&C[i][j]);
        __m128 Bkj = _mm_load_ps(&B[k][j]);
        Bkj = _mm_mul_ps(Aik, Bkj);
        Cij = _mm_add_ps(Bkj, Cij);
        _mm_store_ps(&C[i][j], Cij);
      }
    }
  }
  toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[i][j]);
    }
  }
  printf("error: %f\n",err/N/N);
  printPAPI();
  for (int i=0; i<N; i++) {
    delete[] A[i];
    delete[] B[i];
    delete[] C[i];
  }
  delete[] A;
  delete[] B;
  delete[] C;
}
