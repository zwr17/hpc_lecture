#include <cstdio>

__global__ void kernel(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

int main(void) {
  int n = 2048;
  int size = n * sizeof(float);
  float *a, *b = (float*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<n/1024,1024>>>(a);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%f\n",b[i]);
  cudaFree(a);
  free(b);
  return 0;
}