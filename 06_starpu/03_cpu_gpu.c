#include <starpu.h>
#include <math.h>

extern void cpu_codelet(void *descr[], void *_args);
extern void cuda_codelet(void *descr[], void *_args);

typedef void (*device_func)(void **, void *);

int execute_on(uint32_t where, device_func func, float *block, int pnx, int pny, int pnz, float multiplier)
{
  struct starpu_codelet cl;
  starpu_data_handle_t block_handle;
  int i;

  starpu_vector_data_register(&block_handle, STARPU_MAIN_RAM, (uintptr_t)block, pnx*pny*pnz, sizeof(float));

  starpu_codelet_init(&cl);
  cl.where = where;
  cl.cuda_funcs[0] = func;
  cl.cpu_funcs[0] = func;
  cl.nbuffers = 1;
  cl.modes[0] = STARPU_RW,
    cl.model = NULL;
  cl.name = "block_scale";

  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  task->callback_func = NULL;
  task->handles[0] = block_handle;
  task->cl_arg = &multiplier;
  task->cl_arg_size = sizeof(multiplier);

  int ret = starpu_task_submit(task);

  starpu_task_wait_for_all();

  /* update the array in RAM */
  starpu_data_unregister(block_handle);

  for(i=0 ; i<pnx*pny*pnz; i++) {
    printf("%f ", block[i]);
  }
  printf("\n");

  return 0;
}

int main(void)
{
  float *block, n=1.0;
  int i, j, k, ret;
  int nx=3;
  int ny=2;
  int nz=4;
  float multiplier=1.0;

  ret = starpu_init(NULL);

  block = (float*)malloc(nx*ny*nz*sizeof(float));
  ret = execute_on(STARPU_CPU, cpu_codelet, block, nx, ny, nz, 1.0);
  ret = execute_on(STARPU_CUDA, cuda_codelet, block, nx, ny, nz, 3.0);

  free(block);

  starpu_shutdown();
}
