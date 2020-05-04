#include <starpu.h>

static __global__ void cuda_block(float *a, int N) {
  for(int i=0; i<N; i++)
    a[i] = 3*i;
}

extern "C" void cuda_codelet(void *buffer[], void *_args) {
  float *a = (float *)STARPU_VECTOR_GET_PTR(buffer[0]);
  int N = STARPU_VECTOR_GET_NX(buffer[0]);
  cuda_block<<<1,1,0,starpu_cuda_get_local_stream()>>>(a,N);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
