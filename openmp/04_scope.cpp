#include <iostream>
#include <omp.h>

int main() {
  int i = 0;
#pragma omp parallel
  i = omp_get_thread_num();
  std::cout << i << std::endl;
}
