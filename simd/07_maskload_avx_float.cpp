#include <cstdio>
#include <immintrin.h>

int main() {
  float a[8];
  for (int i=0; i<8; i++) a[i] = i + 1;
  __m256 b = _mm256_load_ps(a);
  __m256 three = _mm256_set1_ps(3);
  __m256 mask = _mm256_cmp_ps(b, three, _CMP_GT_OQ);
  __m256i maski = _mm256_castps_si256(mask);
  b = _mm256_maskload_ps(a, maski);
  _mm256_store_ps(a, b);
  for (int i=0; i<8; i++) printf("%f\n",a[i]);
}
