#include <iostream>

int main() {
  const int N = 16;
  float a[N], b = 0;
  for(int i=0; i<N; i++)
    a[i] = 1;
  for(int i=0; i<N; i++)
    b += a[i];
  std::cout << b << std::endl;
}
