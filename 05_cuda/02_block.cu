#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

int main(void) {
  int N = 2048;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  kernel<<<N/1024,1024>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}