#include <cstdio>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float a[N], b[N];
  for(int i=0; i<N; i++)
    a[i] = i * M_PI / (N - 1);
  __m256 avec = _mm256_load_ps(a);
  avec = _mm256_sin_ps(avec);
  _mm256_store_ps(b, avec);
  for(int i=0; i<N; i++)
    printf("%g %g\n",a[i],b[i]);
}
