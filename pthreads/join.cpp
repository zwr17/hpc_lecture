#include <pthread.h>
#include <stdio.h>

static int i=0;

void* print(void*) {
  printf("%d\n", i++);
}

int main() {
  for(int t=0; t<10; t++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, NULL);
    pthread_join(thread, NULL);
  }
  pthread_exit(NULL);
}
