#include <stdio.h>

void print() {
  static int t=0;
  t++;
#pragma omp barrier
  printf("%d\n", t);
}

int main() {
#pragma omp parallel for
  for(int i=0; i<10; i++) {
    print();
  }
}
