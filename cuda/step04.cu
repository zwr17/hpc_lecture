#include <cstdio>

__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

#define N 512

int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  for (int i=0; i<N; i++) {
    a[i] = 2;
    b[i] = 7;
  }

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  add<<<1,N>>>(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++)
    printf("%d %d\n",i, c[i]);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}