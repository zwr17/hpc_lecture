#include <starpu.h>

int main() {
  int err = starpu_init(NULL);
  printf("%d CPU cores\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
  printf("%d CUDA GPUs\n", starpu_worker_get_count_by_type(STARPU_CUDA_WORKER));
  starpu_shutdown();
}
