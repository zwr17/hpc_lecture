#include <thrust/device_vector.h>
#include <iostream>

int main(void) {
  thrust::device_vector<int> a(4);
  thrust::sequence(a.begin(), a.end());
  for(int i=0; i<a.size(); i++)
    std::cout << a[i] << std::endl;
}