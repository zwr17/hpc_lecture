#include <cstdio>
#include <zmmintrin.h>

int main() {
  float a[16];
  for (int i=0; i<16; i++) a[i] = i + 1;
  __m512 b = _mm512_load_ps(a);
  b = _mm512_add_ps(b, b);
  _mm512_store_ps(a, b);
  for (int i=0; i<16; i++) printf("%f\n",a[i]);
}
