#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
using namespace std;
typedef vector<vector<float>> matrix;

constexpr int ALIGN = 32;
constexpr int SIMD = (256 / sizeof(float) / 8);

void matmult(const float *A, const float *B, float *C, int N) {
  // Intel Xeon E5-2680 v4
  // L1 Cache: 32 KiB
  // L2 Cache: 256 KiB
  // L3 Cache: 1,280 KiB per thread
  const int m = N, n = N, k = N;
  const int kc = 512;
  const int nc = 64;
  const int mc = 256;
  const int nr = 64;
  const int mr = 32;
  alignas(ALIGN) float Ac[mc*kc];
  alignas(ALIGN) float Bc[kc*nc];
  alignas(ALIGN) float Cc[mc*nc];
#pragma omp parallel for collapse(2) private(Ac,Bc,Cc)
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int p=0; p<kc; p++) {
	memcpy(Bc + p * nc, B + (p + pc) * N + jc,
	       nc * sizeof(float));
      }
      for (int ic=0; ic<m; ic+=mc) {
	for (int i=0; i<mc; i++) {
	  memcpy(Ac + i * kc, A + (i + ic) * N + pc,
		 kc * sizeof(float));
	  for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
	}
	for (int jr=0; jr<nc; jr+=nr) {
	  for (int ir=0; ir<mc; ir+=mr) {
	    for (int kr=0; kr<kc; kr++) {
	      for (int i=ir; i<ir + mr; i++) {
		const auto Ac_p =
		  _mm256_broadcast_ss(Ac + i * kc + kr);
		for (int j=jr; j<jr + nr; j+=SIMD) {
		  const auto Bc_p =
		    _mm256_load_ps(Bc + kr * nc + j);
		  auto Cc_p=_mm256_load_ps(Cc + i * nc + j);
		  _mm256_store_ps(
				  Cc + i * nc + j,
				  _mm256_fmadd_ps(Ac_p, Bc_p, Cc_p));
		}
	      }
	    }
	  }
	}
	for (int i=0; i<mc; i++) {
	  for (int j=0; j<nc; j+=SIMD) {
	    const auto C_p =
	      _mm256_load_ps(C + (i + ic) * N + jc + j);
	    const auto Cc_p=_mm256_load_ps(Cc + i * nc + j);

	    _mm256_store_ps(C + (i + ic) * N + jc + j,
			    _mm256_add_ps(C_p, Cc_p));
	  }
	}
      }
    }
  }
}

// ./a.out M N K
int main(int argc, char *argv[]) {
  int N = 4096;
  float *A = new float [N*N];
  float *B = new float [N*N];
  float *C = nullptr;
  posix_memalign(reinterpret_cast<void **>(&C), ALIGN,
		 N * N * sizeof(float));
  matmult(A,B,C,N);
  for (int i=0; i<N*N; i++) {
    A[i] = drand48();
    B[i] = drand48();
    C[i] = 0;
  }
  auto tic = chrono::steady_clock::now();
  matmult(A,B,C,N);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
	C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err+=fabs(C[N*i+j]);
  printf("error: %lf\n",err/N/N);
  free(A);
  free(B);
  free(C);
}
