#include <cstdlib>
#include <cstdio>
#include <cmath>
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
  const int m = N, n = N, k = N;
  const int kc = 512;
  const int nc = 512;
  const int mc = 512;
  const int nr = 32;
  const int mr = 32;
  float Ac[mc][kc];
  float Bc[kc][nc];
  float Cc[mc][nc];
#pragma omp parallel for
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p][j] = B[p+pc][j+jc];
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i][p] = A[i+ic][p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i][j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                for (int j=jr; j<jr+nr; j++) { 
                  Cc[i][j] += Ac[i][kr] * Bc[kr][j];
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[i+ic][j+jc] += Cc[i][j];
          }
        }
      }
    }
  }
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[i][j] -= A[i][k] * B[k][j];
      }
    }
  }
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
