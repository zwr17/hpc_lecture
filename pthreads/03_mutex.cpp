#include <pthread.h>
#include <stdio.h>

void* print(void*) {
  static int i=0;
  static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  printf("%d\n", i);
  i++;
  pthread_mutex_unlock(&mutex);
}

int main() {
  for(int t=0; t<10; t++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, NULL);
  }
  pthread_exit(NULL);
}
