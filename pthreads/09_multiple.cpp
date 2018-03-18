#include <stdio.h>
#include <pthread.h>

const int end=-1;
const int loops=2;
const int producers=2;
const int consumers=2;
const int num_threads=producers+consumers;
int value=0;
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t full=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void print_headers(int producers, int consumers) {
  printf("   ");
  for (int i=0; i<producers; i++)
    printf(" P%d    ", i);
  for (int i=0; i<consumers; i++)
    printf(" C%d    ", i);
  printf("\n");
}

void print(int thread_id, const char *str) {
  printf("[");
  if (value == end) {
    printf("%s", "E");
  } else {
    printf("%d", value);
  }
  printf("] ");
  for (int i=0; i<thread_id; i++) {
    printf("       ");
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
  int id=(size_t) arg;
  int base=id*loops+1;
  for (int i=0; i<loops; i++) {
    pthread_mutex_lock(&mutex); print(id, "lock  ");
    while (value != 0) { print(id, "full  "); print(id, "unlock");
      pthread_cond_wait(&empty, &mutex); print(id, "resume"); print(id, "lock  ");
    }
    put(base+i); print(id, "put   ");
    pthread_cond_signal(&full); print(id, "unlock");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

void *consumer(void *arg) {
  int id=(size_t) arg;
  int tmp=0;
  while (tmp != end) {
    pthread_mutex_lock(&mutex); print(id, "lock  ");
    while (value == 0) { print(id, "empty "); print(id, "unlock");
      pthread_cond_wait(&full, &mutex); print(id, "resume"); print(id, "lock  ");
    }
    tmp=get(); print(id, "get   ");
    pthread_cond_signal(&empty); print(id, "unlock");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  print_headers(producers, consumers);
  pthread_t pid[num_threads], cid[num_threads];
  size_t thread_id=0;
  for (int i=0; i<producers; i++) {
    pthread_create(&pid[i], NULL, producer, (void *) thread_id);
    thread_id++;
  }
  for (int i=0; i<consumers; i++) {
    pthread_create(&cid[i], NULL, consumer, (void *) thread_id);
    thread_id++;
  }
  for (int i=0; i<producers; i++) {
    pthread_join(pid[i], NULL);
  }
  for (int i=0; i<consumers; i++) {
    pthread_mutex_lock(&mutex);
    while (value != 0)
      pthread_cond_wait(&empty, &mutex);
    put(end);
    pthread_cond_signal(&full);
    pthread_mutex_unlock(&mutex);
  }
  for (int i=0; i<consumers; i++) {
    pthread_join(cid[i], NULL);
  }
  return 0;
}
