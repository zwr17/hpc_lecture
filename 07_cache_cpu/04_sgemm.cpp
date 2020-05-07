#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace std;

extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);

int main(int argc, char **argv) {
  int N = 2048;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  auto tic = chrono::steady_clock::now();
  float alpha = 1.0;
  float beta = 0.0;
  char T = 'N';
  sgemm_(&T, &T, &N, &N, &N, &alpha, &B[0], &N, &A[0], &N, &beta, &C[0], &N);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  tic = chrono::steady_clock::now();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
      }
    }
  }
  toc = chrono::steady_clock::now();
  time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[N*i+j]);
    }
  }
  printf("error: %f\n",err/N/N);
}
