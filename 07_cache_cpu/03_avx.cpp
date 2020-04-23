#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>
#include <sys/time.h>

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
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      __m256 Aik = _mm256_broadcast_ss(&A[i][k]);
      for (int j=0; j<N; j+=8) {
        __m256 Cij = _mm256_loadu_ps(&C[i][j]);
        __m256 Bkj = _mm256_loadu_ps(&B[k][j]);
        Cij = _mm256_fmadd_ps(Aik, Bkj, Cij);
        _mm256_storeu_ps(&C[i][j], Cij);
      }
    }
  }
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  gettimeofday(&tic, NULL);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[i][j] -= A[i][k] * B[k][j];
      }
    }
  }
  gettimeofday(&toc, NULL);
  time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[i][j]);
    }
  }
  printf("error: %f\n",err/N/N);
  for (int i=0; i<N; i++) {
    delete[] A[i];
    delete[] B[i];
    delete[] C[i];
  }
  delete[] A;
  delete[] B;
  delete[] C;
}
