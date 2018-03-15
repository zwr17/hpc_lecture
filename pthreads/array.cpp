#include <pthread.h>
#include <stdio.h>

static int i=0;

void* print(void* arg) {
  double* a = (double*)arg;
  printf("%d %lf\n", i++, a[i]);
}

int main() {
  double* a = new double [10];
  for(int t=0; t<10; t++) a[t]=t*10;
  for(int t=0; t<10; t++) {
    pthread_t thread;
    pthread_create(&thread, NULL, print, (void*)a);
  }
  pthread_exit(NULL);
  delete[] a;
}
