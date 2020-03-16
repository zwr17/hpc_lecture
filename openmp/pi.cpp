#include <iostream>
#include <iomanip>

int main() {
  int n = 10;
  double dx = 1. / n;
  double pi = 0;
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  std::cout << std::setprecision(16) << pi << std::endl;
}
