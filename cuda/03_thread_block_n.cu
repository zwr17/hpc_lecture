#include <cstdio>

__global__ void kernel(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = i;
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a;
  cudaMallocManaged(&a, size);
  kernel<<<(n+m-1)/m,m>>>(a,n);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) printf("%f\n",a[i]);
  cudaFree(a);
  return 0;
}