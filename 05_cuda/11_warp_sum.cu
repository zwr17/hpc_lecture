#include <cstdio>

__device__
int warpSum(float sum) {
  for (int offset=16; offset>0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  return sum;
}

__global__ void kernel(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = warpSum(a[i]);
  if ((threadIdx.x & 31) == 0) atomicAdd(&a[0], sum);
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  for (int i=0; i<n; i++) b[i] = 1;
  cudaMemcpy(a, b, size, cudaMemcpyHostToDevice);
  kernel<<<(n+m-1)/m,m>>>(a, n);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  printf("%f\n",b[0]);
  cudaFree(a);
  free(b);
  return 0;
}
