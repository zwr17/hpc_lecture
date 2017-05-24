#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  float ** Ap = new float * [N];
  float ** Bp = new float * [N];
  float ** Cp = new float * [N];
  float * A = new float [N*N];
  float * B = new float [N*N];
  float * C = new float [N*N];
  for (int i=0; i<N; i++) {
    Ap[i] = new float [N];
    Bp[i] = new float [N];
    Cp[i] = new float [N];
    for (int j=0; j<N; j++) {
      Ap[i][j] = drand48();
      Bp[i][j] = drand48();
      Cp[i][j] = 0;
      A[N*i+j] = Ap[i][j];
      B[N*i+j] = Bp[i][j];
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
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        Cp[i][j] += Ap[i][k] * Bp[k][j];
      }
    }
  }
  toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[N*i+j]-Cp[i][j]);
    }
  }
  printf("error: %f\n",err/N/N);
  for (int i=0; i<N; i++) {
    delete[] Ap[i];
    delete[] Bp[i];
    delete[] Cp[i];
  }
  delete[] Ap;
  delete[] Bp;
  delete[] Cp;
  delete[] A;
  delete[] B;
  delete[] C;
}
