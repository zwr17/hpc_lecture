#include <stdio.h>

void print() {
  static int t=0;
  printf("%d\n", t);
  t++;
}

int main() {
#pragma omp parallel for
  for(int i=0; i<10; i++) {
    print();
  }
}
