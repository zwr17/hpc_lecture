#include <iostream>

int main() {
  int a[8] = {2,1,3,2,1,1,1,2};
  int b[8];
#pragma omp parallel
  for(int j=1; j<=4; j<<=1) {
#pragma omp for
    for(int i=0; i<8; i++)
      b[i] = a[i];
#pragma omp for
    for(int i=j; i<8; i++)
      a[i] += b[i-j];
  }
  for(int i=0; i<8; i++)
    std::cout << a[i] << " ";
  std::cout << std::endl;
}
