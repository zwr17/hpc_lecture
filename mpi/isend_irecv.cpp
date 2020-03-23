#include "mpi.h"
#include <cmath>
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int send[4] = {0,0,0,0}, recv[4] = {0,0,0,0};
  for(int i=0; i<4; i++)
    send[i] = mpirank+10*i;
  int send_rank = (mpirank + 1) % mpisize;
  int recv_rank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Request request[2];
  MPI_Status status[2];
  MPI_Irecv(recv, 4, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &request[0]);
  MPI_Isend(send, 4, MPI_INT, send_rank, 0, MPI_COMM_WORLD, &request[1]);
  MPI_Waitall(2, request, status);
  printf("rank%d: send=[%d %d %d %d], recv=[%d %d %d %d]\n",mpirank,
         send[0],send[1],send[2],send[3],recv[0],recv[1],recv[2],recv[3]);
  MPI_Finalize();
}
