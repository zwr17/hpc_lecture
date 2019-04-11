#include "mpi.h"
#include <cstdio>

struct mystruct {
  char a;
  int b;
  double c;
};

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int size[3] = {1,1,1};
  MPI_Aint addr[3], disp[3];
  MPI_Datatype type[3] = {MPI_CHAR, MPI_INT, MPI_DOUBLE}, mytype;
  mystruct send, recv;
  MPI_Address(&send.a, &addr[0]);
  MPI_Address(&send.b, &addr[1]);
  MPI_Address(&send.c, &addr[2]);
  for (int i=0; i<3; i++) disp[i] = addr[i] - addr[0];
  MPI_Type_struct(3, size, disp, type, &mytype);
  MPI_Type_commit(&mytype);
  send.a = 'a';
  send.b = mpirank;
  send.c = 0.1 * (mpirank + 1);
  int send_rank = (mpirank + 1) % mpisize;
  int recv_rank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Send(&send, 1, mytype, send_rank, 0, MPI_COMM_WORLD);
  MPI_Recv(&recv, 1, mytype, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printf("rank%d: send=[%c %d %lf], recv=[%c %d %lf]\n",mpirank,
         send.a,send.b,send.c,recv.a,recv.b,recv.c);
  MPI_Finalize();
}
