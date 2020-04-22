#include <cstdio>
#include <openacc.h>

int main() {
#pragma acc parallel loop gang num_gangs(8) vector_length(1)
  for(int i=0; i<2; i++) {
#pragma acc loop vector
    for(int j=0; j<4; j++) {
      printf("%d: %d, %d: %d\n",
             __pgi_gangidx(),i,
             __pgi_vectoridx(),j);
    }
  }
}
