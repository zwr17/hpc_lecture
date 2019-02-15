#include <cstdio>

__global__ void kernel(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  int n = 4;
  int size = n * sizeof(float);
  float *a;
  cudaMallocManaged(&a, size);
  kernel<<<1,n>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) printf("%f\n",a[i]);
  cudaFree(a);
  return 0;
}