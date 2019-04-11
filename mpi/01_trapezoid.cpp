#include <stdio.h>
#include "mpi.h"

int main( int argc, char *argv[] ) {
  int size, rank, n, err;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if( rank==0 ) {
    printf("Enter the number of intervals\n");
    err = scanf("%d", &n );
  }
  MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD );
  const double h = 2.0/n;

  double sum = 0.0, pi = 0.0;
  for( int i=rank; i<n; i+=size ) {
    const double x0 = i*h - 1.0;
    const double x1 = (i+1)*h - 1.0;
    const double f0 = 2.0 / ( 1.0 + x0*x0 );
    const double f1 = 2.0 / ( 1.0 + x1*x1 );
    sum += 0.5*( f0 + f1 )*( x1 - x0 );
  }
  MPI_Finalize();
  return 0;
}
