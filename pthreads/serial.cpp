#include <stdio.h>

static int i=0;

void print() {
  printf("%d\n", i++);
}

int main() {
  for(int t=0; t<10; t++) {
    print();
  }
}
