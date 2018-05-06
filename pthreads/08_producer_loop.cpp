#include <stdio.h>
#include <pthread.h>

const int end=-1;
const int loops=2;
int value=0;
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t full=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void print(int thread_id, const char *str) {
  printf("[");
  if (value == end) {
    printf("%s", "E");
  } else {
    printf("%d", value);
  }
  printf("] ");
  for (int i=0; i<thread_id; i++) {
    printf("        ");
  }
  printf("%6s\n", str);
}

void put(int v) {
  value=v;
}

int get() {
  int tmp=value;
  value=0;
  return tmp;
}

void *producer(void *arg) {
  for (int i=0; i<loops; i++) {
    pthread_mutex_lock(&mutex); print(0, "lock  ");
    while (value != 0) { print(0, "full  "); print(0, "unlock");
      pthread_cond_wait(&empty, &mutex); print(0, "resume"); print(0, "lock  ");
    }
    put(1); print(0, "put   ");
    pthread_cond_signal(&full); print(0, "unlock");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

void *consumer(void *arg) {
  int tmp=0;
  while (tmp != end) {
    pthread_mutex_lock(&mutex); print(1, "lock  ");
    while (value == 0) { print(1, "empty "); print(1, "unlock");
      pthread_cond_wait(&full, &mutex); print(1, "resume"); print(1, "lock  ");
    }
    tmp=get(); print(1, "get   ");
    pthread_cond_signal(&empty); print(1, "unlock");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

int main() {
  printf("    Produce Consume  \n");
  pthread_t pid, cid;
  pthread_create(&pid, NULL, producer, NULL);
  pthread_create(&cid, NULL, consumer, NULL);
  pthread_join(pid, NULL);
  pthread_mutex_lock(&mutex);
  while (value != 0)
    pthread_cond_wait(&empty, &mutex);
  put(end);
  pthread_cond_signal(&full);
  pthread_mutex_unlock(&mutex);
  pthread_join(cid, NULL);
  return 0;
}
