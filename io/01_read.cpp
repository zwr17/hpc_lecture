#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;

int main () {
  struct timeval tic, toc;
  ifstream file("data.bin", ios::binary | ios::ate);
  int N = file.tellg();
  char *buffer = new char [N];
  file.seekg (0, ios::beg);
  gettimeofday(&tic, NULL);
  file.read(buffer, N);
  gettimeofday(&toc, NULL);
  file.close();
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  int sum = 0;
  for (int i=0; i<N; i++) {
    sum += buffer[i] - '0';
    assert(buffer[i] - '0' == 1);
  }
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,N/time/1e9);
  printf("sum=%d\n",sum);
  delete[] buffer;
  return 0;
}
