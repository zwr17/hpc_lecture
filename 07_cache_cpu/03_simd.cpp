#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <immintrin.h>
using namespace std;

int main(int argc, char **argv) {
  const int N = 2048;
  vector<vector<float>> A(N,vector<float>(N));
  vector<vector<float>> B(N,vector<float>(N));
  vector<vector<float>> C(N,vector<float>(N));
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  auto tic = chrono::steady_clock::now();
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
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
}
