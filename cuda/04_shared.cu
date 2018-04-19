#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float b[2];
  __syncthreads();
  b[threadIdx.x] = 10 * blockIdx.x + threadIdx.x;
  __syncthreads();
  a[i] = b[threadIdx.x];
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