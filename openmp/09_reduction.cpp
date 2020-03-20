#include <iostream>

int main() {
  int a = 0;
#pragma omp parallel for reduction(+:a)
  for(int i=0; i<10000; i++) {
    a += 1;
  }
  std::cout << a << std::endl;
}
