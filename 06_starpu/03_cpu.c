#include <starpu.h>

void cpu_codelet(void *buffer[], void *args) {
  float *a = (float *)STARPU_VECTOR_GET_PTR(buffer[0]);
  int N = (int)STARPU_VECTOR_GET_NX(buffer[0]);
  for(int i=0; i<N; i++)
    a[i] = i;
}
extern void cuda_codelet(void *buffer[], void *args);

typedef void (*device_func)(void **, void *);

int execute_on(uint32_t where, device_func func, float *vector, int N) {
  struct starpu_codelet cl;
  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, N, sizeof(float));
  starpu_codelet_init(&cl);
  cl.where = where;
  cl.cuda_funcs[0] = func;
  cl.cpu_funcs[0] = func;
  cl.nbuffers = 1;
  starpu_task_insert(&cl,STARPU_RW,vector_handle,0);
  starpu_task_wait_for_all();
  starpu_data_unregister(vector_handle);
  for(int i=0 ; i<N; i++) {
    printf("%g ", vector[i]);
  }
  printf("\n");
}

int main(void)
{
  float *vector;
  int N=8;
  int ret = starpu_init(NULL);
  vector = (float*)malloc(N*sizeof(float));
  ret = execute_on(STARPU_CPU, cpu_codelet, vector, N);
  ret = execute_on(STARPU_CUDA, cuda_codelet, vector, N);
  free(vector);
  starpu_shutdown();
}
