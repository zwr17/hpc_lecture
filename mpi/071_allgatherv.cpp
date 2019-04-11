#include "mpi.h"
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int send[4] = {0,0,0,0}, recv[4] = {0,0,0,0};
  int rdsp[3], rcnt[3] = {1,2,1};
  rdsp[0] = 0;
  for(int i=1; i<3; i++) rdsp[i] = rdsp[i-1] + rcnt[i-1];
  rcnt[0] = 2;
  rcnt[1] = 2;
  rcnt[2] = 2;
  send[0] = mpirank+1;
  if(mpirank==1) send[1] = mpirank+1;
  MPI_Allgatherv(send, 2, MPI_INT, recv, rcnt, rdsp, MPI_INT, MPI_COMM_WORLD);
  printf("rank%d: send=[%d %d %d %d], recv=[%d %d %d %d]\n",mpirank,
         send[0],send[1],send[2],send[3],recv[0],recv[1],recv[2],recv[3]);
  MPI_Finalize();
}
