#include <cstdio>
#include <immintrin.h>

int main() {
  double a[4];
  for (int i=0; i<4; i++) a[i] = i + 1;
  __m256d b = _mm256_load_pd(a);
  b = _mm256_add_pd(b, b);
  _mm256_store_pd(a, b);
  for (int i=0; i<4; i++) printf("%lf\n",a[i]);
}
