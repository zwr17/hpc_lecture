#include <pthread.h>
#include <stdio.h>

const int size=1000000;
static double sum=0;

void* print(void* arg) {
  static int t=0;
  static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  t++;
  pthread_mutex_unlock(&mutex);
  int ibegin = (t-1)*size/10;
  int iend = t*size/10;
  printf("thread %d, range %d - %d\n", t-1, ibegin, iend-1);
  double *a = (double*)arg;
  pthread_mutex_lock(&mutex);
  for (int i=ibegin; i<iend; i++) sum+=a[i];
  pthread_mutex_unlock(&mutex);
}

int main() {
  double *a = new double [size];
  for (int i=0; i<size; i++) a[i] = 0.1;
  pthread_t thread[10];
  for(int i=0; i<10; i++) {
    pthread_create(&thread[i], NULL, print, (void*)a);
  }
  printf("sum = %lf\n", sum);
  for(int i=0; i<10; i++) {
    pthread_join(thread[i], NULL);
  }
  printf("sum = %lf\n", sum);
  delete[] a;
  pthread_exit(NULL);
}
