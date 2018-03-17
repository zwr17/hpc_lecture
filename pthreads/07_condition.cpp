#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

void *Malloc(size_t size) {
  void *p = malloc(size);
  assert(p != NULL);
  return p;
}

void Mutex_init(pthread_mutex_t *m) {
  assert(pthread_mutex_init(m, NULL) == 0);
}

void Mutex_lock(pthread_mutex_t *m) {
  int rc = pthread_mutex_lock(m);
  assert(rc == 0);
}

void Mutex_unlock(pthread_mutex_t *m) {
  int rc = pthread_mutex_unlock(m);
  assert(rc == 0);
}

void Cond_init(pthread_cond_t *c) {
  assert(pthread_cond_init(c, NULL) == 0);
}

void Cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
  int rc = pthread_cond_wait(c, m);
  assert(rc == 0);
}

void Cond_signal(pthread_cond_t *c) {
  int rc = pthread_cond_signal(c);
  assert(rc == 0);
}


void Pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                    void *(*start_routine)(void*), void *arg) {
  int rc = pthread_create(thread, attr, start_routine, arg);
  assert(rc == 0);
}

void Pthread_join(pthread_t thread, void **value_ptr) {
  int rc = pthread_join(thread, value_ptr);
  assert(rc == 0);
}

#define MAX_THREADS (100)  // maximum number of producers/consumers

int producers = 1;         // number of producers
int consumers = 1;         // number of consumers

int *buffer;               // the buffer itself: malloc in main()
int max;                   // size of the producer/consumer buffer

int use_ptr  = 0;          // tracks where next consume should come from
int fill_ptr = 0;          // tracks where next produce should go to
int num_full = 0;          // counts how many entries are full

int loops;                 // number of items that each producer produces

#define EMPTY         (-2) // buffer slot has nothing in it
#define END_OF_STREAM (-1) // consumer who grabs this should exit

// used in producer/consumer signaling protocol
pthread_cond_t empty  = PTHREAD_COND_INITIALIZER;
pthread_cond_t fill   = PTHREAD_COND_INITIALIZER;
pthread_mutex_t m     = PTHREAD_MUTEX_INITIALIZER;

#define p0 do_pause(id, 1, 0, "p0");
#define p1 do_pause(id, 1, 1, "p1");
#define p2 do_pause(id, 1, 2, "p2");
#define p3 do_pause(id, 1, 3, "p3");
#define p4 do_pause(id, 1, 4, "p4");
#define p5 do_pause(id, 1, 5, "p5");
#define p6 do_pause(id, 1, 6, "p6");

#define c0 do_pause(id, 0, 0, "c0");
#define c1 do_pause(id, 0, 1, "c1");
#define c2 do_pause(id, 0, 2, "c2");
#define c3 do_pause(id, 0, 3, "c3");
#define c4 do_pause(id, 0, 4, "c4");
#define c5 do_pause(id, 0, 5, "c5");
#define c6 do_pause(id, 0, 6, "c6");

int producer_pause_times[MAX_THREADS][7];
int consumer_pause_times[MAX_THREADS][7];

// needed to avoid interleaving of print out from threads
pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;

void do_print_headers() {
  int i;
  printf("%3s ", "NF");
  for (i = 0; i < max; i++) {
    printf(" %3s ", "   ");
  }
  printf("  ");

  for (i = 0; i < producers; i++)
    printf("P%d ", i);
  for (i = 0; i < consumers; i++)
    printf("C%d ", i);
  printf("\n");
}

void do_print_pointers(int index) {
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

void do_print_buffer() {
  int i;
  printf("%3d [", num_full);
  for (i = 0; i < max; i++) {
    do_print_pointers(i);
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

void do_eos() {
  Mutex_lock(&print_lock);
  do_print_buffer();
  printf("[main: added end-of-stream marker]\n");
  Mutex_unlock(&print_lock);
}

void do_pause(int thread_id, int is_producer, int pause_slot, char *str) {
  int i;
  Mutex_lock(&print_lock);
  do_print_buffer();
  for (i = 0; i < thread_id; i++) {
    printf("   ");
  }
  printf("%s\n", str);
  Mutex_unlock(&print_lock);

  int local_id = thread_id;
  int pause_time;
  if (is_producer) {
    pause_time = producer_pause_times[local_id][pause_slot];
  } else {
    local_id = thread_id - producers;
    pause_time = consumer_pause_times[local_id][pause_slot];
  }
  // printf(" PAUSE %d\n", pause_time);
  sleep(pause_time);
}

void ensure(int expression, char *msg) {
  if (expression == 0) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
  }
}

void parse_pause_string(char *str, char *name, int expected_pieces,
                        int pause_array[MAX_THREADS][7]) {

  // string looks like this (or should):
  //   1,2,0:2,3,4,5
  //   n-1 colons if there are n producers/consumers
  //   comma-separated for sleep amounts per producer or consumer
  int index = 0;

  char *copy_entire = strdup(str);
  char *outer_marker;
  int colon_count = 0;
  char *p = strtok_r(copy_entire, ":", &outer_marker);
  while (p) {
    // init array: default sleep is 0
    int i;
    for (i = 0; i < 7; i++)
      pause_array[index][i] = 0;

    // for each piece, comma separated
    char *inner_marker;
    char *copy_piece = strdup(p);
    char *c = strtok_r(copy_piece, ",", &inner_marker);
    int comma_count = 0;

    int inner_index = 0;
    while (c) {
      int pause_amount = atoi(c);
      ensure(inner_index < 7, "you specified a sleep string incorrectly... (too many comma-separated args)");
      // printf("setting %s pause %d to %d\n", name, inner_index, pause_amount);
      pause_array[index][inner_index] = pause_amount;
      inner_index++;

      c = strtok_r(NULL, ",", &inner_marker);
      comma_count++;
    }
    free(copy_piece);
    index++;

    // continue with colon separated list
    p = strtok_r(NULL, ":", &outer_marker);
    colon_count++;
  }

  free(copy_entire);
  if (expected_pieces != colon_count) {
    fprintf(stderr, "Error: expected %d %s in sleep specification, got %d\n", expected_pieces, name, colon_count);
    exit(1);
  }
}

void do_fill(int value) {
  // ensure empty before usage
  ensure(buffer[fill_ptr] == EMPTY, "error: tried to fill a non-empty buffer");
  buffer[fill_ptr] = value;
  fill_ptr = (fill_ptr + 1) % max;
  num_full++;
}

int do_get() {
  int tmp = buffer[use_ptr];
  ensure(tmp != EMPTY, "error: tried to get an empty buffer");
  buffer[use_ptr] = EMPTY;
  use_ptr = (use_ptr + 1) % max;
  num_full--;
  return tmp;
}

void *producer(void *arg) {
  int id = (size_t) arg;
  // make sure each producer produces unique values
  int base = id * loops;
  int i;
  for (i = 0; i < loops; i++) {   p0;
    Mutex_lock(&m);             p1;
    while (num_full == max) {   p2;
      Cond_wait(&empty, &m);  p3;
    }
    do_fill(base + i);          p4;
    Cond_signal(&fill);         p5;
    Mutex_unlock(&m);           p6;
  }
  return NULL;
}

void *consumer(void *arg) {
  int id = (size_t) arg;
  int tmp = 0;
  int consumed_count = 0;
  while (tmp != END_OF_STREAM) { c0;
    Mutex_lock(&m);            c1;
    while (num_full == 0) {    c2;
      Cond_wait(&fill, &m);  c3;
    }
    tmp = do_get();            c4;
    Cond_signal(&empty);       c5;
    Mutex_unlock(&m);          c6;
    consumed_count++;
  }

  // return consumer_count-1 because END_OF_STREAM does not count
  return (void *) (long long) (consumed_count - 1);
}

// must set these appropriately to use "main-common.c"
pthread_cond_t *fill_cv = &fill;
pthread_cond_t *empty_cv = &empty;

// Common usage() prints out stuff for command-line help
void usage() {
  fprintf(stderr, "usage: \n");
  fprintf(stderr, "  -l <number of items each producer produces>\n");
  fprintf(stderr, "  -m <size of the shared producer/consumer buffer>\n");
  fprintf(stderr, "  -p <number of producers>\n");
  fprintf(stderr, "  -c <number of consumers>\n");
  fprintf(stderr, "  -P <sleep string: how each producer should sleep at various points in execution>\n");
  fprintf(stderr, "  -C <sleep string: how each consumer should sleep at various points in execution>\n");
  fprintf(stderr, "  -v [ verbose flag: trace what is happening and print it ]\n");
  fprintf(stderr, "  -t [ timing flag: time entire execution and print total time ]\n");
  exit(1);
}

// Common main() for all four programs
// - Does arg parsing
// - Starts producers and consumers
// - Once producers are finished, puts END_OF_STREAM
//   marker into shared queue to signal end to consumers
// - Then waits for consumers and prints some final info
int main(int argc, char *argv[]) {
  loops = 1;
  max = 1;
  consumers = 1;
  producers = 1;

  char *producer_pause_string = NULL;
  char *consumer_pause_string = NULL;

  opterr = 0;
  int c;
  while ((c = getopt (argc, argv, "l:m:p:c:P:C:vt")) != -1) {
    switch (c) {
    case 'l':
      loops = atoi(optarg);
      break;
    case 'm':
      max = atoi(optarg);
      break;
    case 'p':
      producers = atoi(optarg);
      break;
    case 'c':
      consumers = atoi(optarg);
      break;
    case 'P':
      producer_pause_string = optarg;
      break;
    case 'C':
      consumer_pause_string = optarg;
      break;
    default:
      usage();
    }
  }

  assert(loops > 0);
  assert(max > 0);
  assert(producers <= MAX_THREADS);
  assert(consumers <= MAX_THREADS);

  if (producer_pause_string != NULL)
    parse_pause_string(producer_pause_string, "producers", producers, producer_pause_times);
  if (consumer_pause_string != NULL)
    parse_pause_string(consumer_pause_string, "consumers", consumers, consumer_pause_times);

  // make space for shared buffer, and init it ...
  buffer = (int *) Malloc(max * sizeof(int));
  int i;
  for (i = 0; i < max; i++) {
    buffer[i] = EMPTY;
  }

  do_print_headers();

  struct timeval tic, toc;
  gettimeofday(&tic, NULL);

  // start up all threads; order doesn't matter here
  pthread_t pid[MAX_THREADS], cid[MAX_THREADS];
  int thread_id = 0;
  for (i = 0; i < producers; i++) {
    Pthread_create(&pid[i], NULL, producer, (void *) (long long) thread_id);
    thread_id++;
  }
  for (i = 0; i < consumers; i++) {
    Pthread_create(&cid[i], NULL, consumer, (void *) (long long) thread_id);
    thread_id++;
  }

  // now wait for all PRODUCERS to finish
  for (i = 0; i < producers; i++) {
    Pthread_join(pid[i], NULL);
  }

  // end case: when producers are all done
  // - put "consumers" number of END_OF_STREAM's in queue
  // - when consumer sees -1, it exits
  for (i = 0; i < consumers; i++) {
    Mutex_lock(&m);
    while (num_full == max)
      Cond_wait(empty_cv, &m);
    do_fill(END_OF_STREAM);
    do_eos();
    Cond_signal(fill_cv);
    Mutex_unlock(&m);
  }

  // now OK to wait for all consumers
  int counts[consumers];
  for (i = 0; i < consumers; i++) {
    pthread_join(cid[i], (void **)&counts[i]);
  }

  gettimeofday(&toc, NULL);

  printf("\nConsumer consumption:\n");
  for (i = 0; i < consumers; i++) {
    printf("  C%d -> %d\n", i, counts[i]);
  }
  printf("\n");

  printf("Total time: %.2f seconds\n", toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  return 0;
}
