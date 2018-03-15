#include <pthread.h>
#include <stdio.h>

const int size=1000000;

void* print(void*) {
  static int t=0;
  static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  t++;
  pthread_mutex_unlock(&mutex);
  int ibegin = (t-1)*size/10;
  int iend = t*size/10;
  printf("thread %d, range %d - %d\n", t-1, ibegin, iend-1);
}

int main() {
  for(int i=0; i<10; i++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, NULL);
  }
  pthread_exit(NULL);
}
