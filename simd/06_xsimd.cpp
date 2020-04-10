#include <cstdio>
#include "xsimd/xsimd.hpp"

int main() {
  const int N = 8;
  xsimd::batch<float, N> a, b, c;
  for(int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i * 0.1;
    c[i] = 0;
  }
  c = a + b;
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,c[i]);
}
