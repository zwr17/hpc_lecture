#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
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
  const int m = N, n = N, k = N;
  const int kc = 512;
  const int nc = 32;
  const int mc = 256;
  const int nr = 32;
  const int mr = 32;
  float Ac[mc*kc];
  float Bc[kc*nc];
  float Cc[mc*nc];
#pragma omp parallel for collapse(2) private(Ac,Bc,Cc)
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p*nc+j] = B[p+pc][j+jc];
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = A[i+ic][p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                for (int j=jr; j<jr+nr; j++) { 
                  Cc[i*nc+j] += Ac[i*kc+kr] * Bc[kr*nc+j];
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[i+ic][j+jc] += Cc[i*nc+j];
          }
        }
      }
    }
  }
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
}
