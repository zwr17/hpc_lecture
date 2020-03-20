#include <iostream>
#include "mpi.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  std::cout << "rank: " << mpirank << "/" << mpisize << std::endl;
  MPI_Finalize();
}
