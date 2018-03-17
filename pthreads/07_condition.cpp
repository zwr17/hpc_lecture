#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

#define MAX_THREADS (100)  // maximum number of producers/consumers

int producers = 1;         // number of producers
int consumers = 1;         // number of consumers
int *buffer;               // the buffer itself: malloc in main()
int max;                   // size of the producer/consumer buffer
int use_ptr  = 0;          // tracks where next consume should come from
int fill_ptr = 0;          // tracks where next produce should go to
int num_full = 0;          // counts how many entries are full

#define EMPTY         (-2) // buffer slot has nothing in it
#define END_OF_STREAM (-1) // consumer who grabs this should exit

// used in producer/consumer signaling protocol
pthread_cond_t empty  = PTHREAD_COND_INITIALIZER;
pthread_cond_t fill   = PTHREAD_COND_INITIALIZER;
pthread_mutex_t m     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;

void print_headers() {
  int i;
  printf("%3s ", "NF");
  for (i = 0; i < max; i++) {
    printf(" %3s ", "   ");
  }
  printf("   ");

  for (i = 0; i < producers; i++)
    printf("P%d ", i);
  for (i = 0; i < consumers; i++)
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
  int i;
  printf("%3d [", num_full);
  for (i = 0; i < max; i++) {
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
  int i;
  pthread_mutex_lock(&print_lock);
  print_buffer();
  for (i = 0; i < thread_id; i++) {
    printf("   ");
  }
  printf("%s\n", str);
  pthread_mutex_unlock(&print_lock);
  sleep(0);
}

void put(int value) {
  buffer[fill_ptr] = value;
  fill_ptr = (fill_ptr + 1) % max;
  num_full++;
}

int get() {
  int tmp = buffer[use_ptr];
  buffer[use_ptr] = EMPTY;
  use_ptr = (use_ptr + 1) % max;
  num_full--;
  return tmp;
}

void *producer(void *arg) {
  const int loops = 4;
  int id = (size_t) arg;
  // make sure each producer produces unique values
  int base = id * loops;
  int i;
  for (i = 0; i < loops; i++) { pause(id, 1, 0, "p0");
    pthread_mutex_lock(&m); pause(id, 1, 1, "p1");
    while (num_full == max) { pause(id, 1, 2, "p2");
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
  int consumed_count = 0;
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

  // return consumer_count-1 because END_OF_STREAM does not count
  return (void *) (long long) (consumed_count - 1);
}

// must set these appropriately to use "main-common.c"
pthread_cond_t *fill_cv = &fill;
pthread_cond_t *empty_cv = &empty;

int main(int argc, char *argv[]) {
  max = 2;
  consumers = 2;
  producers = 1;
  buffer = new int [max];
  for (int i = 0; i < max; i++) {
    buffer[i] = EMPTY;
  }
  print_headers();
  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  pthread_t pid[MAX_THREADS], cid[MAX_THREADS];
  int thread_id = 0;
  for (int i = 0; i < producers; i++) {
    pthread_create(&pid[i], NULL, producer, (void *) (long long) thread_id);
    thread_id++;
  }
  for (int i = 0; i < consumers; i++) {
    pthread_create(&cid[i], NULL, consumer, (void *) (long long) thread_id);
    thread_id++;
  }
  for (int i = 0; i < producers; i++) {
    pthread_join(pid[i], NULL);
  }

  for (int i = 0; i < consumers; i++) {
    pthread_mutex_lock(&m);
    while (num_full == max)
      pthread_cond_wait(empty_cv, &m);
    put(END_OF_STREAM);
    eos();
    pthread_cond_signal(fill_cv);
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
