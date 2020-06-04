#if TIME
#include <time.h>
time_t start;
time_t elapse = 0;
#elif CLOCK
#include <time.h>
clock_t start;
clock_t elapse = 0;
#elif GETTIMEOFDAY
#include <sys/time.h>
struct timeval start; 
struct timeval stop;
double elapse = 0;
#elif BOOST
#include <boost/timer/timer.hpp>
boost::timer::cpu_timer timer;
#elif CHRONO
#include <chrono>
using namespace std::chrono::steady_clock;
time_point start;
time_point stop;
#else
#error Define macro for timer type
#endif

void startTimer() {
#if TIME
  start = time(NULL);
#elif CLOCK
  start = clock();
#elif GETTIMEOFDAY
  gettimeofday(&start, NULL);
#elif BOOST
  timer.start();
#endif
}

void stopTimer() {
#if TIME
  elapse += time(NULL) - start;
#elif CLOCK
  elapse += clock() - start;
#elif GETTIMEOFDAY
  gettimeofday(&stop, NULL);
  elapse += stop.tv_sec-start.tv_sec+(stop.tv_usec-start.tv_usec)*1e-6;
#elif BOOST
  timer.stop();
#endif
}

double getTime() {
#if TIME
  return (double) elapse;
#elif CLOCK
  return (double) elapse / CLOCKS_PER_SEC;
#elif GETTIMEOFDAY
  return elapse;
#elif BOOST
  return (double)timer.elapsed().wall*1e-9;
#endif
}
