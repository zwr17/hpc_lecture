#include <cstdio>
#include <starpu.h>

struct params {
  int i;
  float f;
};

void cpu_func(void *buffers[], void *cl_arg) {
  (void)buffers;
  struct params *params = (struct params *) cl_arg;
  printf("hello\n");
}

int main(void) {
  struct params params = {1, 2.0f};
  int ret = starpu_init(NULL);
  struct starpu_task *task = starpu_task_create();
  struct starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = cpu_func;
  cl.nbuffers = 0;
  task->cl = &cl;
  task->cl_arg = &params;
  task->cl_arg_size = sizeof(params);
  task->synchronous = 1;
  ret = starpu_task_submit(task);
  starpu_shutdown();
}
