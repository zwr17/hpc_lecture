#include <cstdio>

__global__ void kernel(float *a, int n, float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  *sum += a[i];
  //atomicAdd(sum, a[i]);
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a, *sum;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&sum, sizeof(float));
  for (int i=0; i<n; i++) a[i] = 1;
  kernel<<<(n+m-1)/m,m>>>(a, n, sum);
  cudaDeviceSynchronize();
  printf("%f\n",*sum);
  cudaFree(a);
  cudaFree(sum);
  return 0;
}
