#include <stdio.h>
#include <pthread.h>

int value=0;
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t fill=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void print(int thread_id, const char *str) {
  printf("[");
  printf("%d", value);
  printf("] ");
  for (int i=0; i<thread_id; i++) {
    printf("        ");
  }
  printf("%6s\n", str);
}

void put() {
  value=1;
}

void get() {
  value=0;
}

void *producer(void *arg) {
  pthread_mutex_lock(&mutex); print(0, "lock  ");
  put(); print(0, "put   ");
  pthread_cond_signal(&fill); print(0, "unlock");
  pthread_mutex_unlock(&mutex);
  return NULL;
}

void *consumer(void *arg) {
  pthread_mutex_lock(&mutex); print(1, "lock  ");
  while (value == 0) { print(1, "empty "); print(1, "unlock");
    pthread_cond_wait(&fill, &mutex); print(1, "resume"); print(1, "lock  ");
  }
  get(); print(1, "get   "); print(1, "unlock");
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(int argc, char *argv[]) {
  printf("    Produce Consume\n");
  pthread_t pid, cid;
  pthread_create(&pid, NULL, producer, NULL);
  pthread_create(&cid, NULL, consumer, NULL);
  pthread_join(pid, NULL);
  pthread_join(cid, NULL);
  return 0;
}
