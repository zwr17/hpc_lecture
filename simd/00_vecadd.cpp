#include <cstdio>

int main() {
  const int N = 16;
  float a[N], b[N], c[N];
  for(int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i * 0.1;
    c[i] = 0;
  }
  for(int i=0; i<N; i++)
    c[i] = a[i] + b[i];
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,c[i]);
}
