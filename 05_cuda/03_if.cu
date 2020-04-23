#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i % 2)
    a[i] = i;
  else
    a[i] = -i;
}

int main(void) {
  int N = 2048;
  int M = 1024;
  float *a;
  cudaMallocManaged(&a, N*sizeof(float));
  kernel<<<(N+M-1)/M,M>>>(a);
  cudaDeviceSynchronize();
  for (int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cudaFree(a);
}