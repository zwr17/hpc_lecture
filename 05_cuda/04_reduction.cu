#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[0] += a[i];
  //atomicAdd(&a[0], a[i]);
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
