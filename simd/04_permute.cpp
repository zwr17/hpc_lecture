#include <iostream>

int main() {
  const int N = 16;
  int idx[N] = {3,7,12,0,15,6,10,2,14,11,5,8,1,9,4,13};
  float a[N], b[N];
  for(int i=0; i<N; i++) {
    a[idx[i]] = i * 0.1;
  }
  for(int i=0; i<N; i++)
    b[i] = a[idx[i]];
  for(int i=0; i<N; i++)
    std::cout << i << " " << b[i] << std::endl;
}
