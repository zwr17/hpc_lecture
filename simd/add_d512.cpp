#include <cstdio>
#include <immintrin.h>

int main() {
  double a[8];
  for (int i=0; i<8; i++) a[i] = i + 1;
  __m512d b = _mm512_load_pd(a);
  b = _mm512_add_pd(b, b);
  _mm512_store_pd(a, b);
  for (int i=0; i<8; i++) printf("%lf\n",a[i]);
}
