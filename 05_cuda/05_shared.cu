#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float b[1024];
  __syncthreads();
  b[threadIdx.x] = a[i];
  __syncthreads();
  float c = 0;
  for (int j=0; j<blockDim.x; j++)
    c += b[j];
  if ((threadIdx.x & (blockDim.x-1)) == 0)
    atomicAdd(&a[0], c);
}

int main(void) {
  int N = 2047;
  int M = 1024;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  for (int i=0; i<N; i++) a[i] = 1;
  kernel<<<(N+M-1)/M,M>>>(a);
  cudaDeviceSynchronize();
  printf("%g\n",a[0]);
  cudaFree(a);
}