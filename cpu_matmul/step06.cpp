#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  float * A = new float [N*N];
  float * B = new float [N*N];
  float * C = new float [N*N];
  float * D = new float [N*N];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  double tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[N*i+j] += A[N*i+k] * B[N*k+j];
      }
    }
  }
  double toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  tic = get_time();
  float alpha = 1.0;
  float beta = 0.0;
  char T = 'N';
  sgemm_(&T,&T, &N, &N, &N, &alpha, B, &N, A, &N, &beta, D, &N);
  toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[N*i+j]-D[N*i+j]);
    }
  }
  printf("error: %f\n",err/N/N);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
}
