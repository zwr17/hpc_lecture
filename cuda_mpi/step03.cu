#include "mpi.h"
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int send[4] = {0,0,0,0}, recv[4] = {0,0,0,0};
  for(int i=0; i<4; i++)
    send[i] = mpirank+10*i;
  int sendrank = (mpirank + 1) % mpisize;
  int recvrank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Send(send, 4, MPI_INT, sendrank, 0, MPI_COMM_WORLD);
  MPI_Recv(recv, 4, MPI_INT, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int irank=0; irank<mpisize; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (irank == mpirank) {
      printf("rank%d: send_rank=%d, recv_rank=%d\n", mpirank, sendrank, recvrank);
      printf("send=[%d %d %d %d], recv=[%d %d %d %d]\n",
             send[0],send[1],send[2],send[3],recv[0],recv[1],recv[2],recv[3]);
    }
  }
  MPI_Finalize();
}
