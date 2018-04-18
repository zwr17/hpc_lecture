#include <cstdio>

__global__ void kernel(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  int size = 4 * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<1,4>>>(a);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<4; i++) printf("%f\n",b[i]);
  cudaFree(a);
  free(b);
  return 0;
}