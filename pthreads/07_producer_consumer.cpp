#include <stdio.h>
#include <pthread.h>

#define EMPTY -2           // buffer slot has nothing in it
#define END -1             // consumer who grabs this should exit
int value=EMPTY;           // the value
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t fill=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

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
  pthread_mutex_lock(&mutex); print(0, "lock");
  put(0); print(0, "put ");
  pthread_cond_signal(&fill); print(0, "uloc");
  pthread_mutex_unlock(&mutex);
  return NULL;
}

void *consumer(void *arg) {
  int tmp=EMPTY;
  pthread_mutex_lock(&mutex); print(1, "lock");
  while (value == EMPTY) { print(1, "none"); print(1, "uloc");
    pthread_cond_wait(&fill, &mutex); print(1, "resu"); print(1, "lock");
  }
  tmp=get(); print(1, "get "); print(1, "uloc");
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(int argc, char *argv[]) {
  printf("    Prod Cons\n");
  pthread_t pid, cid;
  pthread_create(&pid, NULL, producer, NULL);
  pthread_create(&cid, NULL, consumer, NULL);
  pthread_join(pid, NULL);
  pthread_join(cid, NULL);
  return 0;
}
