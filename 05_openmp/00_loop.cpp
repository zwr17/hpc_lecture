#include <cstdio>
#include <omp.h>

int main() {
  const int N=8;
  int *a = new int [N];
#pragma omp target map(tofrom:a[0:N])
  {
#pragma omp parallel for
    for(int i=0; i<N; i++)
      a[i] = omp_get_thread_num();
  }
  for(int i=0; i<N; i++)
    printf("%d: %d\n",a[i],i);
  delete[] a;
}
