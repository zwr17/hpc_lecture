#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

#define EMPTY         (-2) // buffer slot has nothing in it
#define END_OF_STREAM (-1) // consumer who grabs this should exit
#define MAX_THREADS 100    // maximum number of producers/consumers
#define MAX_BUFFER 1       // maximum capacity of buffer
int *buffer;               // the buffer itself: malloc in main()
int use_ptr=0;             // tracks where next consume should come from
int fill_ptr=0;            // tracks where next produce should go to
int count=0;               // counts how many entries are full
pthread_cond_t empty=PTHREAD_COND_INITIALIZER;
pthread_cond_t fill=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

void print_headers(int producers, int consumers) {
  printf("%2s ", "N");
  for (int i=0; i<MAX_BUFFER; i++) {
    printf(" %s ", " ");
  }
  printf("  ");
  for (int i=0; i<producers; i++)
    printf(" P%d  ", i);
  for (int i=0; i<consumers; i++)
    printf(" C%d  ", i);
  printf("\n");
}

void print(int thread_id, int is_producer, int pause_slot, const char *str) {
  printf("%d [", count);
  for (int i=0; i<MAX_BUFFER; i++) {
    if (use_ptr == i && fill_ptr == i) {
      printf("*");
    } else if (use_ptr == i) {
      printf("u");
    } else if (fill_ptr == i) {
      printf("f");
    } else {
      printf(" ");
    }
    if (buffer[i] == EMPTY) {
      printf("%s ", "-");
    } else if (buffer[i] == END_OF_STREAM) {
      printf("%s ", "E");
    } else {
      printf("%d ", buffer[i]);
    }
  }
  printf("] ");
  for (int i=0; i<thread_id; i++) {
    printf("     ");
  }
  printf("%4s\n", str);
  sleep(0);
}

void put(int value) {
  buffer[fill_ptr]=value;
  fill_ptr=(fill_ptr + 1) % MAX_BUFFER;
  count++;
}

int get() {
  int tmp=buffer[use_ptr];
  buffer[use_ptr]=EMPTY;
  use_ptr=(use_ptr + 1) % MAX_BUFFER;
  count--;
  return tmp;
}

void *producer(void *arg) {
  const int loops=2;
  int id=(size_t) arg;
  int base=id * loops;
  for (int i=0; i<loops; i++) {
    pthread_mutex_lock(&mutex); print(id, 1, 1, "lock");
    while (count == MAX_BUFFER) { print(id, 1, 2, "full"); print(id, 1, 2, "uloc");
      pthread_cond_wait(&empty, &mutex); print(id, 1, 3, "resu"); print(id, 1, 3, "lock");
    }
    put(base + i); print(id, 1, 4, "put ");
    pthread_cond_signal(&fill); print(id, 1, 5, "uloc");
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}

void *consumer(void *arg) {
  int id=(size_t) arg;
  int tmp=0;
  size_t consumed_count=0;
  while (tmp != END_OF_STREAM) {
    pthread_mutex_lock(&mutex); print(id, 0, 1, "lock");
    while (count == 0) { print(id, 0, 2, "empt"); print(id, 0, 2, "uloc");
      pthread_cond_wait(&fill, &mutex); print(id, 0, 3, "resu"); print(id, 0, 3, "lock");
    }
    tmp=get(); print(id, 0, 4, "get ");
    pthread_cond_signal(&empty); print(id, 0, 5, "uloc");
    pthread_mutex_unlock(&mutex);
    consumed_count++;
  }
  return (void *) (consumed_count - 1);
}

int main(int argc, char *argv[]) {
  int producers=2;
  int consumers=2;
  buffer=new int [MAX_BUFFER];
  for (int i=0; i<MAX_BUFFER; i++) {
    buffer[i]=EMPTY;
  }
  print_headers(producers, consumers);
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  pthread_t pid[MAX_THREADS], cid[MAX_THREADS];
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
    while (count == MAX_BUFFER)
      pthread_cond_wait(&empty, &mutex);
    put(END_OF_STREAM);
    pthread_cond_signal(&fill);
    pthread_mutex_unlock(&mutex);
  }

  int counts[consumers];
  for (int i=0; i<consumers; i++) {
    pthread_join(cid[i], (void **)&counts[i]);
  }
  gettimeofday(&toc, NULL);
  printf("\nConsumer consumption:\n");
  for (int i=0; i<consumers; i++) {
    printf("  C%d -> %d\n", i, counts[i]);
  }
  printf("\n");
  printf("Total time: %.2f seconds\n", toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  return 0;
}
