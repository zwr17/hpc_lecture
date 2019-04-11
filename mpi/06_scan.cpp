#include "mpi.h"
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int send[2], recv[2] = {0};
  for (int i=0; i<4; i++) send[i] = mpirank + i;
  MPI_Scan(send, recv, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  printf("rank%d: send=[%d %d], recv=[%d %d]\n",
         mpirank,send[0],send[1],recv[0],recv[1]);
  MPI_Finalize();
}
