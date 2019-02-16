#include <cstdio>

struct Record {
  int *key;
  int *value;
  int *flag;
};

__global__ void kernel(Record a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a.key[i] = i;
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(int);
  Record a, b;
  b.key = (int*) malloc(size);
  b.value = (int*) malloc(size);
  b.flag = (int*) malloc(size);
  cudaMalloc(&a.key, size);
  cudaMalloc(&a.value, size);
  cudaMalloc(&a.flag, size);
  kernel<<<(n+m-1)/m,m>>>(a,n);
  cudaMemcpy(b.key, a.key, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.value, a.value, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.flag, a.flag, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%d\n",b.key[i]);
  cudaFree(a.key);
  cudaFree(a.value);
  cudaFree(a.flag);
  free(b.key);
  free(b.value);
  free(b.flag);
  return 0;
}