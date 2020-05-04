#include <starpu.h>

void cpu_codelet(void *buffer[], void *args) {
  float *a = (float *)STARPU_VECTOR_GET_PTR(buffer[0]);
  int N = (int)STARPU_VECTOR_GET_NX(buffer[0]);
  for(int i=0; i<N; i++)
    a[i] = i; 
}

