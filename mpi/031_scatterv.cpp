#include "mpi.h"
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int send[4] = {0,0,0,0}, recv[4] = {0,0,0,0};
  int sdsp[3], scnt[3] = {1,2,1};
  sdsp[0] = 0;
  for (int i=1; i<3; i++) sdsp[i] = sdsp[i-1] + scnt[i-1];
  if(!mpirank) {
    for(int i=0; i<4; i++) send[i] = i+1;
  }
  MPI_Scatterv(send, scnt, sdsp, MPI_INT, recv, 2, MPI_INT, 0, MPI_COMM_WORLD);
  printf("rank%d: send=[%d %d %d %d], recv=[%d %d %d %d]\n",mpirank,
         send[0],send[1],send[2],send[3],recv[0],recv[1],recv[2],recv[3]);
  MPI_Finalize();
}
