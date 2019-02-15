#include <cstdio>

__global__ void kernel(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  int n = 4;
  int size = n * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<1,n>>>(a);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%f\n",b[i]);
  cudaFree(a);
  free(b);
  return 0;
}