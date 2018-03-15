#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>

const size_t size=1000000000;
const int nthreads=4;
static size_t sum=0;
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond=PTHREAD_COND_INITIALIZER;

void* print(void* arg) {
  static int t=0;
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
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
}

int main() {
  size_t *a = new size_t [size];
  for (size_t i=0; i<size; i++) a[i] = 1;
  pthread_t thread[nthreads];
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  for(int i=0; i<nthreads; i++) {
    printf("create: %d\n",i);
    pthread_create(&thread[i], NULL, print, (void*)a);
  }
  pthread_mutex_lock(&mutex);
  while (sum < size) pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  gettimeofday(&toc, NULL);
  printf("sum = %ld\n", sum);
  printf("%lf s\n",toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  delete[] a;
  pthread_exit(NULL);
}
