#include <cstdio>

__global__ void kernel(float *a) {
  a[threadIdx.x] = threadIdx.x;
}

int main(void) {
  int size = 4 * sizeof(float);
  float *a;
  cudaMallocManaged(&a, size);
  kernel<<<1,4>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<4; i++) printf("%f\n",a[i]);
  cudaFree(a);
  return 0;
}