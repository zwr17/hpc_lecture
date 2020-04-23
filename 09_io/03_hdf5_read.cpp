#include <iostream>
#include <sys/time.h>
#include "H5Cpp.h"
using namespace H5;

int main (int argc, char** argv) {
  struct timeval tic, toc;
  H5File file("data.h5", H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet("name");
  DataSpace dataspace = dataset.getSpace();
  int ndim = dataspace.getSimpleExtentNdims();
  hsize_t dim[ndim];
  dataspace.getSimpleExtentDims(dim);
  int N = 1;
  for (int i=0; i<ndim; i++) N *= dim[i]; 
  int *buffer = new int [N];
  gettimeofday(&tic, NULL);
  dataset.read(buffer, PredType::NATIVE_INT);
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  int sum = 0;
  for (int i=0; i<N; i++) {
    sum += buffer[i];
  }
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("sum=%d\n",sum);
  delete[] buffer;
  return 0;
}
