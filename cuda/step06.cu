#include <cstdio>

__global__ void add(int *a, int *b, int *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}

#define N (2048*2048+43)
#define M 512

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

  add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  int sum_c = 0;
  for (int i=0; i<N; i++)
    sum_c += c[i];
  printf("%d %d\n",sum_c,N*9);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}