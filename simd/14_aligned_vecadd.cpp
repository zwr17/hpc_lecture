#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

int main() {
  const int N = 8;
  float *a = (float*)aligned_alloc(32, N*sizeof(float));
  float *b = (float*)aligned_alloc(32, N*sizeof(float));
  float *c = (float*)aligned_alloc(32, N*sizeof(float));
  for(int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i * 0.1;
    c[i] = 0;
  }
  asm volatile ("# begin loop");
  for(int i=0; i<N; i++)
    c[i] = a[i] + b[i];
  asm volatile ("# end loop");
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,c[i]);
  free(a);
  free(b);
  free(c);
}
