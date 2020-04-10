#include <cstdio>
#include <cmath>

int main() {
  const int N = 16;
  float a[N], b[N];
  for(int i=0; i<N; i++)
    a[i] = i * M_PI / (N - 1);
  for(int i=0; i<N; i++)
    b[i] = sin(a[i]);
  for(int i=0; i<N; i++)
    printf("%g %g\n",a[i],b[i]);
}
