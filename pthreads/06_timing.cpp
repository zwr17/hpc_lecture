#include <pthread.h>
#include <stdio.h>

const size_t size=1000000000;
const int nthreads=1;
static size_t sum=0;

void* print(void* arg) {
  static int t=0;
  static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  t++;
  pthread_mutex_unlock(&mutex);
  size_t ibegin = (t-1)*size/nthreads;
  size_t iend = t*size/nthreads;
  size_t *a = (size_t*)arg;
  size_t partial=0;
  for (size_t i=ibegin; i<iend; i++) partial+=a[i];
  pthread_mutex_lock(&mutex);
  sum += partial;
  pthread_mutex_unlock(&mutex);
}

int main() {
  size_t *a = new size_t [size];
  for (size_t i=0; i<size; i++) a[i] = 1;
  pthread_t thread[nthreads];
  for(int i=0; i<nthreads; i++) {
    pthread_create(&thread[i], NULL, print, (void*)a);
  }
  printf("sum = %ld\n", sum);
  for(int i=0; i<nthreads; i++) {
    pthread_join(thread[i], NULL);
  }
  printf("sum = %ld\n", sum);
  delete[] a;
  pthread_exit(NULL);
}
