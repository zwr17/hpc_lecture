#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace std;

int main() {
  const int N = 1024;
  vector<vector<float>> A(N,vector<float>(N));
  vector<vector<float>> B(N,vector<float>(N));
  vector<vector<float>> C(N,vector<float>(N));
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  auto tic = chrono::steady_clock::now();
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[i][j] += A[i][k] * B[k][j];
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
}
