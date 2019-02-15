#include <cstdio>

__device__
int warpSum(float sum) {
  for (int offset=16; offset>0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  return sum;
}

__global__ void kernel(float *a, float *sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float b = warpSum(a[i]);
  if ((threadIdx.x & 31) == 0) atomicAdd(sum, b);
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a, *sum;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&sum, sizeof(float));
  for (int i=0; i<n; i++) a[i] = 1;
  kernel<<<(n+m-1)/m,m>>>(a, sum, n);
  cudaDeviceSynchronize();
  printf("%f\n",*sum);
  cudaFree(a);
  cudaFree(sum);
  return 0;
}
