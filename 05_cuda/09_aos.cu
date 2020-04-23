#include <cstdio>

struct Record {
  int key;
  int value;
  int flag;
};

__global__ void kernel(Record *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i].key = i;
}

int main(void) {
  int n = 2047;
  int m = 1024;
  int size = n * sizeof(Record);
  Record *a, *b = (Record*) malloc(size);
  cudaMalloc(&a, size);
  kernel<<<(n+m-1)/m,m>>>(a,n);
  cudaMemcpy(b, a, size, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%d\n",b[i].key);
  cudaFree(a);
  free(b);
  return 0;
}