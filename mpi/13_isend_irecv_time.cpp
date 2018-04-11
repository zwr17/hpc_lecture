#include "mpi.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int main(int argc, char ** argv) {
  struct timeval tic, toc;
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int size = atoi(argv[1]);
  int * send = new int [size];
  int * recv = new int [size];
  for(int i=0; i<size; i++)
    send[i] = mpirank+size*i;
  int send_rank = (mpirank + 1) % mpisize;
  int recv_rank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Request request[2];
  MPI_Status status[2];
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&tic, NULL);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Irecv(recv, 4, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, &request[0]);
  MPI_Isend(send, 4, MPI_INT, send_rank, 0, MPI_COMM_WORLD, &request[1]);
  MPI_Waitall(2, request, status);
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&toc, NULL);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%lf s\n",toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  delete[] send;
  delete[] recv;
  MPI_Finalize();
}
