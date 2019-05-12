#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <cblas.h>
#include <lapacke.h>

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  double * A = new double [N*N];
  double * x = new double [N];
  double * b = new double [N];
  int * ipiv = new int [N];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = 1 / (std::abs(i - j) + 1e-3);
    }
    x[i] = drand48();
  }
  for (int i=0; i<N; i++) {
    b[i] = 0;
    for (int j=0; j<N; j++) {
      b[i] += A[N*i+j] * x[j];
    }
  }
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A, N, ipiv);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1, A, N, b, 1);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1, A, N, b, 1);
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s\n",N,time);
  double err = 0;
  for (int i=0; i<N; i++) {
    err += fabs(b[i]-x[i]);
  }
  printf("error: %f\n",err/N/N);
  delete[] A;
  delete[] x;
  delete[] b;
  delete[] ipiv;
}
