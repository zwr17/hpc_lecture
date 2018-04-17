#include <cstdio>

__global__ void kernel(float *a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = 10 * blockIdx.x + threadIdx.x;
}

int main(void) {
  const int size = 4 * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<2,2>>>(a);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<4; i++) printf("%f\n",b[i]);
  cudaFree(a);
  free(b);
  return 0;
}