#include <stdio.h>
#include <pthread.h>

#define EMPTY -2           // buffer slot has nothing in it
#define END -1             // consumer who grabs this should exit
const int loops=2;         // number of producer loops
const int producers=1;     // number of producers
const int consumers=1;     // number of consumers
const int num_threads=2;
int value=EMPTY;           // the value
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t fill=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void print_headers() {
}

void print(int thread_id, const char *str) {
  printf("[");
  if (value == EMPTY) {
    printf("%s", "-");
  } else if (value == END) {
    printf("%s", "E");
  } else {
    printf("%d", value);
  }
  printf("] ");
  for (int i=0; i<thread_id; i++) {
    printf("     ");
  }
  printf("%4s\n", str);
}

void put(int v) {
  value=v;
}

int get() {
  int tmp=value;
  value=EMPTY;
  return tmp;
}

void *producer(void *arg) {
  for (int i=0; i<loops; i++) {
    pthread_mutex_lock(&mutex); print(0, "lock");
    while (value != EMPTY) { print(0, "full"); print(0, "uloc");
      pthread_cond_wait(&empty, &mutex); print(0, "resu"); print(0, "lock");
    }
    put(i); print(0, "put ");
    pthread_cond_signal(&fill); print(0, "uloc");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

void *consumer(void *arg) {
  int tmp=0;
  while (tmp != END) {
    pthread_mutex_lock(&mutex); print(1, "lock");
    while (value == EMPTY) { print(1, "none"); print(1, "uloc");
      pthread_cond_wait(&fill, &mutex); print(1, "resu"); print(1, "lock");
    }
    tmp=get(); print(1, "get ");
    pthread_cond_signal(&empty); print(1, "uloc");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  printf("    Prod Cons\n");
  pthread_t pid, cid;
  pthread_create(&pid, NULL, producer, NULL);
  pthread_create(&cid, NULL, consumer, NULL);
  pthread_join(pid, NULL);
  pthread_mutex_lock(&mutex);
  while (value != EMPTY)
    pthread_cond_wait(&empty, &mutex);
  put(END);
  pthread_cond_signal(&fill);
  pthread_mutex_unlock(&mutex);
  pthread_join(cid, NULL);
  return 0;
}
