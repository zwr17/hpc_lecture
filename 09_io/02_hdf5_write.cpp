#include <iostream>
#include <chrono>
#include "H5Cpp.h"
using namespace H5;

int main (int argc, char** argv) {
  const int NX = 10000, NY = 10000;
  int *buffer = new int [NX*NY];
  for (int i=0; i<NX; i++) {
    for (int j=0; j<NY; j++) {
      buffer[NY*i+j] = 1;
    }
  }
  H5File file("data.h5", H5F_ACC_TRUNC);
  hsize_t dim[2] = {NX, NY};
  DataSpace dataspace(2,dim);
  DataSet dataset = file.createDataSet("name", PredType::NATIVE_INT, dataspace);
  auto tic = chrono::steady_clock::now();
  dataset.write(buffer, PredType::NATIVE_INT);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  delete[] buffer;
  return 0;
}
