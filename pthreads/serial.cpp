#include <stdio.h>


void print() {
  static int i=0;
  printf("%d\n", i);
  i++;
}

int main() {
  for(int t=0; t<10; t++) {
    print();
  }
}
