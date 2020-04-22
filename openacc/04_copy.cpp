#include <cstdio>

int main(void) {
  float *a = new float [4];
#pragma acc parallel loop copyout(a[:4])
  for (int i=0; i<4; i++) a[i] = i;
  for (int i=0; i<4; i++) printf("%d %g\n",i,a[i]);
  delete[] a;
}
