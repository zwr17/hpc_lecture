#include <iostream>

int fib(int n) {
  if (n<2) return n;
  int i = fib(n-1);
  int j = fib(n-2);
  return i+j;
}

int main() {
  int n = 10;
  for (int i=0; i<n; i++) {
    std::cout << fib(i) << " ";
  }
  std::cout << std::endl;
}
