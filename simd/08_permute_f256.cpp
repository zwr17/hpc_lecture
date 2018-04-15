#include <cstdio>
#include <immintrin.h>

int main() {
  float a[8];
  for (int i=0; i<8; i++) a[i] = i + 1;
  __m256 b = _mm256_load_ps(a);
  b = _mm256_permute_ps(b,0b01110100);
  _mm256_store_ps(a, b);
  for (int i=0; i<8; i++) printf("%f\n",a[i]);
}
