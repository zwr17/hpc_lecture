#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;

int main (int argc, char** argv) {
  const int N = atoi(argv[1]);
  char *buffer = new char [N];
  for (int i=0; i<N; i++) buffer[i] = '1';
  ofstream file("data.bin", ios::binary);

  auto tic = chrono::steady_clock::now();
  file.write(buffer, N);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GB/s\n",N,time,N/time/1e9);
  delete[] buffer;
  return 0;
}
