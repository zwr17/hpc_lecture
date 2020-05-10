#include <cmath>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
using namespace std;

#define M 1024

__global__ void matmul(float *A, float *B, float *C, int N) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  __shared__ float A_s[M];
  for (int ks=0; ks<N; ks+=M) {
    __syncthreads();
    A_s[threadIdx.x] = A[N*i+ks+threadIdx.x];
    __syncthreads();
    for (int k=ks; k<ks+M; k++) {
      sum += A_s[k-ks] * B[N*k+j];
    }
  }
  C[N*i+j] = sum;
}

int main(int argc, char **argv) {
  int N = 2048;
  float * h_A = new float [N*N];
  float * h_B = new float [N*N];
  float * h_C = new float [N*N];
  float * h_D = new float [N*N];
  float *d_A, *d_B, *d_C, *d_D;
  int size = N * N * sizeof(float);
  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);
  cudaMalloc((void **) &d_C, size);
  cudaMalloc((void **) &d_D, size);

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      h_A[N*i+j] = drand48();
      h_B[N*i+j] = drand48();
      h_C[N*i+j] = 0;
      h_D[N*i+j] = 0;
    }
  }
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
  dim3 grid(N/M, N);
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  matmul<<<grid,M>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  float alpha = 1.0;
  float beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  stat = cublasSetMatrix(N, N, sizeof(*h_A), h_A, N, d_A, N);
  stat = cublasSetMatrix(N, N, sizeof(*h_B), h_B, N, d_B, N);
  stat = cublasSetMatrix(N, N, sizeof(*h_D), h_D, N, d_D, N);
  gettimeofday(&tic, NULL);
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, d_B, N, d_A, N, &beta, d_D, N);
  stat = cublasGetMatrix(N, N, sizeof(*h_D), d_D, N, h_D, N);
  gettimeofday(&toc, NULL);
  time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(h_C[N*i+j]-h_D[N*i+j]);
    }
  }
  printf("error: %f\n",err/N/N);
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        h_D[N*i+j] -= h_A[N*i+k] * h_B[N*k+j];
  err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(h_D[N*i+j]);
  printf("error: %lf\n",err/N/N);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cublasDestroy(handle);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;
}
