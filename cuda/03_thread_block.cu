#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = 10 * blockIdx.x + threadIdx.x;
}

int main(void) {
  int size = 4 * sizeof(float);
  float *a;
  cudaMallocManaged(&a, size);
  kernel<<<2,2>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<4; i++) printf("%f\n",a[i]);
  cudaFree(a);
  return 0;
}