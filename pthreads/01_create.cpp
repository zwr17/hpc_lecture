#include <pthread.h>
#include <stdio.h>

void* print(void*) {
  static int i=0;
  printf("%d\n", i);
  i++;
}

int main() {
  for(int t=0; t<10; t++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, NULL);
  }
  pthread_exit(NULL);
}
