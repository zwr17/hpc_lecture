#include <cstdio>
#include <cuda_runtime.h>
#include <starpu.h>

static __global__ void vectors(int N, float *vector, float *ptr) {
  int i = i = blockIdx.x*blockDim.x + threadIdx.x;
  a[i] =  i;
}

extern "C" void vectors(void *buffers[], void *) {
  struct starpu_vector_interface *vector_handle =
    (struct starpu_vector_interface *)buffers[0];
  int N = STARPU_VECTOR_GET_NX(vector_handle);
  float *a = (float*)STARPU_VECTOR_GET_PTR(vector_handle);
  float *ptr = cl_arg;
  int threads = 64;
  int blocks = (N+threads-1)/threads;
  vectors<<<blocks,threads,0,starpu_cuda_get_local_stream ()>>>(N,a,ptr);
}

int main(void) {
  const int N = 8;
  float a[N];
  int ret = starpu_init(NULL);
  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle,0,(uintptr_t)a,N,sizeof(float));
  struct starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = vectors;
  cl.nbuffers = 1;
  starpu_task_insert(&cl,STARPU_RW,vector_handle,0);
  starpu_task_wait_for_all();
  starpu_data_unregister(vector_handle);
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  starpu_shutdown();
}
