#include <pthread.h>
#include <stdio.h>

void *print(void* arg) {
  printf("%ld\n", size_t(arg));
}

int main() {
  const int num_threads = 10;
  pthread_t threads[num_threads];
  for(size_t t=0; t<num_threads; t++) {
    pthread_create(&threads[t], NULL, print, (void *)t);
  }
  pthread_exit(NULL);
}
