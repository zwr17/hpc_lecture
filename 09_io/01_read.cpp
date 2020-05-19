#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <chrono>
using namespace std;

int main () {
  ifstream file("data.bin", ios::binary | ios::ate);
  int N = file.tellg();
  char *buffer = new char [N];
  file.seekg (0, ios::beg);
  auto tic = chrono::steady_clock::now();
  file.read(buffer, N);
  auto toc = chrono::steady_clock::now();
  file.close();
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<N; i++) {
    sum += buffer[i] - '0';
  }
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,N/time/1e9);
  printf("sum=%d\n",sum);
  delete[] buffer;
  return 0;
}
