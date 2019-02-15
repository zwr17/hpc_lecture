#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

int main(void) {
  int n = 2048;
  int size = n * sizeof(float);
  float *a;
  cudaMallocManaged(&a, size);
  kernel<<<n/1024,1024>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) printf("%f\n",a[i]);
  cudaFree(a);
  return 0;
}