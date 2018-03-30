#include <pthread.h>
#include <stdio.h>

void* print(void*) {
  static int t=0;
  printf("%d\n", t++);
}

int main() {
  for(int i=0; i<10; i++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, NULL);
  }
  pthread_exit(NULL);
}
