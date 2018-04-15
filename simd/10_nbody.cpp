#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

int main() {
// Initialize
  int N = 1 << 16;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  struct timeval tic, toc;
  float * x = (float*) malloc(N * sizeof(float));
  float * y = (float*) malloc(N * sizeof(float));
  float * z = (float*) malloc(N * sizeof(float));
  float * m = (float*) malloc(N * sizeof(float));
  float * p = (float*) malloc(N * sizeof(float));
  float * ax = (float*) malloc(N * sizeof(float));
  float * ay = (float*) malloc(N * sizeof(float));
  float * az = (float*) malloc(N * sizeof(float));
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// No SSE
  gettimeofday(&tic, NULL);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    for (int j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float dz = z[j] - zi;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
  gettimeofday(&toc, NULL);
  double diff = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
  printf("No SIMD: %e s : %lf GFlops\n", diff, OPS/diff);

// DEALLOCATE
  free(x);
  free(y);
  free(z);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(az);
  return 0;
}
