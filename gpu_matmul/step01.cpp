#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>
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
  float * h_A = new float [N*N];
  float * h_B = new float [N*N];
  float * h_C = new float [N*N];
  for (int i=0; i<N; i++) {
    A[i] = new float [N];
    B[i] = new float [N];
    C[i] = new float [N];
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
      h_A[N*i+j] = A[i][j];
      h_B[N*i+j] = B[i][j];
      h_C[N*i+j] = 0;
    }
  }
  double tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        h_C[N*i+j] += h_A[N*i+k] * h_B[N*k+j];
      }
    }
  }
  double toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(h_C[N*i+j]-C[i][j]);
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
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}
