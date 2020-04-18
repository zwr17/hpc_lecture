#include <cstdio>
#include <openacc.h>

int main() {
#pragma acc parallel loop gang
  for(int i=0; i<2; i++) {
    for(int i=0; i<5; i++) {
      printf("%d %d %d: %d\n",
             __pgi_vectoridx(),
             __pgi_workeridx(),
             __pgi_gangidx(),i);
    }
  }
}
