#include <cstdio>
#include <immintrin.h>

int main() {
  float a[8];
  for (int i=0; i<8; i++) a[i] = i + 1;
  __m256 b = _mm256_load_ps(a);
  __m256 c = _mm256_permute2f128_ps(b,b,1);
  c = _mm256_add_ps(c,b);
  c = _mm256_hadd_ps(c,c);
  c = _mm256_hadd_ps(c,c);
  _mm256_store_ps(a, c);
  for (int i=0; i<8; i++) printf("%f\n",a[i]);
}
