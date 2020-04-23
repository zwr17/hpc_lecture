#include <cstdio>

__device__
int warpSum(float sum) {
  for (int offset=16; offset>0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  return sum;
}

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = warpSum(a[i]);
  if ((threadIdx.x & 31) == 0) atomicAdd(&a[0], sum);
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
