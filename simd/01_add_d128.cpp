#include <cstdio>
#include <xmmintrin.h>

int main() {
  double a[2];
  for (int i=0; i<2; i++) a[i] = i + 1;
  __m128d b = _mm_load_pd(a);
  b = _mm_add_pd(b, b);
  _mm_store_pd(a, b);
  for (int i=0; i<2; i++) printf("%lf\n",a[i]);
}
