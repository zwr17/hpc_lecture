#include <cstdio>
#include <xmmintrin.h>

int main() {
  float a[4];
  for (int i=0; i<4; i++) a[i] = i + 1;
  __m128 b = _mm_load_ps(a);
  b = _mm_add_ps(b, b);
  _mm_store_ps(a, b);
  for (int i=0; i<4; i++) printf("%f\n",a[i]);
}
