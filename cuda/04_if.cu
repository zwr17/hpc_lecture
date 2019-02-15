#include <cstdio>

__global__ void kernel(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i % 2)
    a[i] = i;
  else
    a[i] = -i;
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<(n+m-1)/m,m>>>(a,n);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%f\n",b[i]);
  cudaFree(a);
  free(b);
  return 0;
}