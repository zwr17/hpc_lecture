#include <mpi.h>
#include <iostream>
#include <sys/time.h>

int main(int argc, char** argv) {
  const int N = 100000000;
  struct timeval tic, toc;
  int mpisize, mpirank;
  MPI_File file;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int Nlocal = N / mpisize;
  int offset = Nlocal * mpirank;
  int * buffer = new int [Nlocal];
  for(int i=0; i<Nlocal; i++) buffer[i] = 1;
  MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  gettimeofday(&tic, NULL);
  MPI_File_write_at(file, offset, buffer, Nlocal, MPI_INT, MPI_STATUS_IGNORE);
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  MPI_File_close(&file);
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  MPI_Finalize();
  return 0;
}
