#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>

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

// SSE
  gettimeofday(&tic, NULL);
#pragma omp parallel for
  for (int i=0; i<N; i+=4) {
    __m128 pi = _mm_setzero_ps();
    __m128 axi = _mm_setzero_ps();
    __m128 ayi = _mm_setzero_ps();
    __m128 azi = _mm_setzero_ps();
    __m128 xi = _mm_load_ps(x+i);
    __m128 yi = _mm_load_ps(y+i);
    __m128 zi = _mm_load_ps(z+i);
    for (int j=0; j<N; j++) {
      __m128 R2 = _mm_set1_ps(EPS2);
      __m128 x2 = _mm_set1_ps(x[j]);
      x2 = _mm_sub_ps(x2, xi);
      __m128 y2 = _mm_set1_ps(y[j]);
      y2 = _mm_sub_ps(y2, yi);
      __m128 z2 = _mm_set1_ps(z[j]);
      z2 = _mm_sub_ps(z2, zi);
      __m128 xj = x2;
      x2 = _mm_mul_ps(x2, x2);
      R2 = _mm_add_ps(R2, x2);
      __m128 yj = y2;
      y2 = _mm_mul_ps(y2, y2);
      R2 = _mm_add_ps(R2, y2);
      __m128 zj = z2;
      z2 = _mm_mul_ps(z2, z2);
      R2 = _mm_add_ps(R2, z2);
      __m128 mj = _mm_set1_ps(m[j]);
      __m128 invR = _mm_rsqrt_ps(R2);
      mj = _mm_mul_ps(mj, invR);
      pi = _mm_add_ps(pi, mj);
      invR = _mm_mul_ps(invR, invR);
      invR = _mm_mul_ps(invR, mj);
      xj = _mm_mul_ps(xj, invR);
      axi = _mm_add_ps(axi, xj);
      yj = _mm_mul_ps(yj, invR);
      ayi = _mm_add_ps(ayi, yj);
      zj = _mm_mul_ps(zj, invR);
      azi = _mm_add_ps(azi, zj);
    }
    _mm_store_ps(p+i, pi);
    _mm_store_ps(ax+i, axi);
    _mm_store_ps(ay+i, ayi);
    _mm_store_ps(az+i, azi);
  }
  gettimeofday(&toc, NULL);
  double diff = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
  printf("SSE    : %e s : %lf GFlops\n", diff, OPS/diff);

// No SSE
  gettimeofday(&tic, NULL);
  float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
#pragma omp parallel for reduction(+: pdiff, pnorm, adiff, anorm)
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
    pdiff += (p[i] - pi) * (p[i] - pi);
    pnorm += pi * pi;
    adiff += (ax[i] - axi) * (ax[i] - axi)
      + (ay[i] - ayi) * (ay[i] - ayi)
      + (az[i] - azi) * (az[i] - azi);
    anorm += axi * axi + ayi * ayi + azi * azi;
  }
  gettimeofday(&toc, NULL);
  diff = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
  printf("No SIMD: %e s : %lf GFlops\n", diff, OPS/diff);
  printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
  printf("A ERR  : %e\n",sqrt(adiff/anorm));

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
