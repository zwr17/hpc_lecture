#include <cstdio>
#include <starpu.h>

struct Params {
  int i;
  float f;
};

void cpu_func(void **, void *args) {
  Params *p = (Params *) args;
  printf("%d %g\n",p->i,p->f);
}

int main(void) {
  Params p = {1, 1.1};
  int ret = starpu_init(NULL);
  struct starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = cpu_func;
  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  task->cl_arg = &p;
  task->cl_arg_size = sizeof(Params);
  ret = starpu_task_submit(task);
  starpu_shutdown();
}
