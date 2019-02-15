#include <cstdio>

__global__ void kernel(float *a, float *sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __shared__ float b[1024];
  __syncthreads();
  b[threadIdx.x] = a[i];
  __syncthreads();
  float c = 0;
  for (int j=0; j<blockDim.x; j++)
    c += b[j];
  if ((threadIdx.x & (blockDim.x-1)) == 0) atomicAdd(sum, c);
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a, *sum;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&sum, sizeof(float));
  for (int i=0; i<n; i++) a[i] = 1;
  kernel<<<(n+m-1)/m,m>>>(a,sum,n);
  cudaDeviceSynchronize();
  printf("%f\n",*sum);
  cudaFree(a);
  cudaFree(sum);
  return 0;
}