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
#define MAX_BUFFER 2       // maximum capacity of buffer
int *buffer;               // the buffer itself: malloc in main()
int use_ptr  = 0;          // tracks where next consume should come from
int fill_ptr = 0;          // tracks where next produce should go to
int num_full = 0;          // counts how many entries are full


// used in producer/consumer signaling protocol
pthread_cond_t empty  = PTHREAD_COND_INITIALIZER;
pthread_cond_t fill   = PTHREAD_COND_INITIALIZER;
pthread_mutex_t m     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;

void print_headers(int producers, int consumers) {
  printf("%3s ", "NF");
  for (int i = 0; i < MAX_BUFFER; i++) {
    printf(" %3s ", "   ");
  }
  printf("   ");

  for (int i = 0; i < producers; i++)
    printf("P%d ", i);
  for (int i = 0; i < consumers; i++)
    printf("C%d ", i);
  printf("\n");
}

void print_pointers(int index) {
  if (use_ptr == index && fill_ptr == index) {
    printf("*");
  } else if (use_ptr == index) {
    printf("u");
  } else if (fill_ptr == index) {
    printf("f");
  } else {
    printf(" ");
  }
}

void print_buffer() {
  printf("%3d [", num_full);
  for (int i = 0; i < MAX_BUFFER; i++) {
    print_pointers(i);
    if (buffer[i] == EMPTY) {
      printf("%3s ", "---");
    } else if (buffer[i] == END_OF_STREAM) {
      printf("%3s ", "EOS");
    } else {
      printf("%3d ", buffer[i]);
    }
  }
  printf("] ");
}

void eos() {
  pthread_mutex_lock(&print_lock);
  print_buffer();
  printf("[main: added end-of-stream marker]\n");
  pthread_mutex_unlock(&print_lock);
}

void pause(int thread_id, int is_producer, int pause_slot, const char *str) {
  pthread_mutex_lock(&print_lock);
  print_buffer();
  for (int i = 0; i < thread_id; i++) {
    printf("   ");
  }
  printf("%s\n", str);
  pthread_mutex_unlock(&print_lock);
  sleep(0);
}

void put(int value) {
  buffer[fill_ptr] = value;
  fill_ptr = (fill_ptr + 1) % MAX_BUFFER;
  num_full++;
}

int get() {
  int tmp = buffer[use_ptr];
  buffer[use_ptr] = EMPTY;
  use_ptr = (use_ptr + 1) % MAX_BUFFER;
  num_full--;
  return tmp;
}

void *producer(void *arg) {
  const int loops = 4;
  int id = (size_t) arg;
  int base = id * loops;
  for (int i = 0; i < loops; i++) { pause(id, 1, 0, "p0");
    pthread_mutex_lock(&m); pause(id, 1, 1, "p1");
    while (num_full == MAX_BUFFER) { pause(id, 1, 2, "p2");
      pthread_cond_wait(&empty, &m); pause(id, 1, 3, "p3");
    }
    put(base + i); pause(id, 1, 4, "p4");
    pthread_cond_signal(&fill); pause(id, 1, 5, "p5");
    pthread_mutex_unlock(&m); pause(id, 1, 6, "p6");
  }
  return NULL;
}

void *consumer(void *arg) {
  int id = (size_t) arg;
  int tmp = 0;
  size_t consumed_count = 0;
  while (tmp != END_OF_STREAM) { pause(id, 0, 0, "c0");
    pthread_mutex_lock(&m); pause(id, 0, 1, "c1");
    while (num_full == 0) { pause(id, 0, 2, "c2");
      pthread_cond_wait(&fill, &m); pause(id, 0, 3, "c3");
    }
    tmp = get(); pause(id, 0, 4, "c4");
    pthread_cond_signal(&empty); pause(id, 0, 5, "c5");
    pthread_mutex_unlock(&m); pause(id, 0, 6, "c6");
    consumed_count++;
  }
  return (void *) (consumed_count - 1);
}

int main(int argc, char *argv[]) {
  int consumers = 2;
  int producers = 1;
  buffer = new int [MAX_BUFFER];
  for (int i = 0; i < MAX_BUFFER; i++) {
    buffer[i] = EMPTY;
  }
  print_headers(producers, consumers);
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  pthread_t pid[MAX_THREADS], cid[MAX_THREADS];
  size_t thread_id = 0;
  for (int i = 0; i < producers; i++) {
    pthread_create(&pid[i], NULL, producer, (void *) thread_id);
    thread_id++;
  }
  for (int i = 0; i < consumers; i++) {
    pthread_create(&cid[i], NULL, consumer, (void *) thread_id);
    thread_id++;
  }
  for (int i = 0; i < producers; i++) {
    pthread_join(pid[i], NULL);
  }

  for (int i = 0; i < consumers; i++) {
    pthread_mutex_lock(&m);
    while (num_full == MAX_BUFFER)
      pthread_cond_wait(&empty, &m);
    put(END_OF_STREAM);
    eos();
    pthread_cond_signal(&fill);
    pthread_mutex_unlock(&m);
  }

  int counts[consumers];
  for (int i = 0; i < consumers; i++) {
    pthread_join(cid[i], (void **)&counts[i]);
  }
  gettimeofday(&toc, NULL);
  printf("\nConsumer consumption:\n");
  for (int i = 0; i < consumers; i++) {
    printf("  C%d -> %d\n", i, counts[i]);
  }
  printf("\n");
  printf("Total time: %.2f seconds\n", toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  return 0;
}
