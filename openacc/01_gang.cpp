#include <cstdio>
#include <cstdlib>

int main(void) {
  int size = 4 * sizeof(float);
  float *a = (float*) malloc(size);
#pragma acc kernels loop gang(2) vector(2)
  for (int i=0; i<4; i++) a[i] = i;
  for (int i=0; i<4; i++) printf("%f\n",a[i]);
  free(a);
  return 0;
}
