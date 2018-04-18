#include <cstdio>

__global__ void kernel(float *a, float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //*sum += a[i];
  atomicAdd(sum, a[i]);
}

int main(void) {
  int size = 4 * sizeof(float);
  float *a, *sum;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&sum, sizeof(float));
  for (int i=0; i<4; i++) a[i] = 1;
  kernel<<<2,2>>>(a, sum);
  cudaDeviceSynchronize();
  printf("%f\n",*sum);
  cudaFree(a);
  cudaFree(sum);
  return 0;
}