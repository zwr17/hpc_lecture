#include <cstdio>
#include <starpu.h>

void hello(void **, void *) {
  printf("hello\n");
}

int main() {
  int err = starpu_init(NULL);
  struct starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = hello;
  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  err = starpu_task_submit(task);
  starpu_shutdown();
}
