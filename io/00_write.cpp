#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;

int main (int argc, char** argv) {
  const int N = atoi(argv[1]);
  struct timeval tic, toc;
  char *buffer = new char [N];
  for (int i=0; i<N; i++) buffer[i] = '1';
  ofstream file("data.bin", ios::binary);
  gettimeofday(&tic, NULL);
  file.write(buffer, N);
  gettimeofday(&toc, NULL);
  file.close();
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,N/time/1e9);
  delete[] buffer;
  return 0;
}
