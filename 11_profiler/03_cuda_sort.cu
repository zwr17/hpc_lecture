#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fillBucket(int* key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void scanBucket(int *bucket, int *offset, int *buffer, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=range) return;
  offset[i] = bucket[i];
  for(int j=1; j<range; j<<=1) {
    buffer[i] = offset[i];
    __syncthreads();
    if(i>=j) offset[i] += buffer[i-j];
    __syncthreads();
  }
  offset[i] -= bucket[i];
}

__global__ void fillKey(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=range) return;
  int j = offset[i];
  for (; bucket[i]>0; bucket[i]--)
    key[j++] = i;
}

int main() {
  int n = 50;
  int m = 4;
  int range = 5;
  int *key, *bucket, *offset, *buffer;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&buffer, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  for (int i=0; i<range; i++)
    bucket[i] = 0;
  fillBucket<<<(n+m-1)/m,m>>>(key, bucket, n);
  scanBucket<<<(range+m-1)/m,m>>>(bucket, offset, buffer, range);
  fillKey<<<1,range>>>(key, bucket, offset, range);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++)
    printf("%d ",key[i]);
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
  cudaFree(buffer);
}
