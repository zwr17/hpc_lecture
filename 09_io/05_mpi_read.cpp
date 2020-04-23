#include <mpi.h>
#include <iostream>
#include <sys/time.h>

int main(int argc, char** argv) {
  struct timeval tic, toc;
  int mpisize, mpirank;
  MPI_File file;
  MPI_Offset filesize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_get_size(file, &filesize);
  int N = filesize / sizeof(int);
  int Nlocal = N / mpisize;
  int offset = Nlocal * mpirank;
  int * buffer = new int [Nlocal];
  gettimeofday(&tic, NULL);
  MPI_File_read_at(file, offset, buffer, Nlocal, MPI_INT, MPI_STATUS_IGNORE);
  gettimeofday(&toc, NULL);
  MPI_File_close(&file);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  int sum = 0;
  for (int i=0; i<Nlocal; i++)
    sum += buffer[i];
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("rank=%d sum=%d\n",mpirank,sum);
  MPI_Finalize();
  return 0;
}
