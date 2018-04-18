#include <cstdio>

__inline__ __device__
int warpSum(float sum) {
  for (int offset=16; offset>0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  return sum;
}

__global__ void kernel(float *a, float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float b = warpSum(a[i]);
  if ((threadIdx.x & 31) == 0) atomicAdd(sum, b);
}

int main(void) {
  int N = 128;
  int threads = 64;
  int size = N * sizeof(float);
  float *a, *sum;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&sum, sizeof(float));
  for (int i=0; i<N; i++) a[i] = 1;
  kernel<<<N/threads,threads>>>(a, sum);
  cudaDeviceSynchronize();
  printf("%f\n",*sum);
  cudaFree(a);
  cudaFree(sum);
  return 0;
}