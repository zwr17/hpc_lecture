#include <random>
#include <vector>

#include <cassert>  // assert
#include <cstdio>   // sscanf, printf
#include <cstdlib>
#include <cstring>  // memset, memcpy

#include <sys/time.h>

#include <immintrin.h>

constexpr size_t ALIGN = 32;
constexpr size_t mc = 256;
constexpr size_t nc = 64;
constexpr size_t kc = 512;
constexpr size_t nr = 64;
constexpr size_t mr = 32;
constexpr size_t SIMD = (256 / sizeof(float) / 8);

inline double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

void check_error(const float *array1, const float *array2, const size_t N) {
  double cum_error = 0.0;
  for (size_t i = 0; i < N; i++) cum_error += fabs(array1[i] - array2[i]);
  printf("Error: %f\n", cum_error / N);
}

void gemm_naive(const float *A, const float *B, float *C, const size_t M,
		const size_t N, const size_t K) {
#pragma omp parallel for
  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      for (size_t n = 0; n < N; n++) {
	C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void gemm_blocking(const float *A, const float *B, float *C, const size_t M,
		   const size_t N, const size_t K) {
  // Intel Xeon E5-2680 v4
  // L1 Cache: 32 KiB
  // L2 Cache: 256 KiB
  // L3 Cache: 1,280 KiB per thread

#pragma omp parallel for collapse(2)
  for (size_t jc = 0; jc < N; jc += nc) {
    for (size_t pc = 0; pc < K; pc += kc) {
      alignas(ALIGN) float Bc[kc * nc];

      for (size_t p = 0; p < kc; p++) {
	memcpy(Bc + p * nc, B + (p + pc) * N + jc,
		 nc * sizeof(float));
      }

      for (size_t ic = 0; ic < M; ic += mc) {
	alignas(ALIGN) float Ac[mc * kc];

	for (size_t i = 0; i < mc; i++) {
	  memcpy(Ac + i * kc, A + (i + ic) * K + pc,
		   kc * sizeof(float));
	}

	alignas(ALIGN) float Cc[mc * nc] = {0.0f};

	for (size_t jr = 0; jr < nc; jr += nr) {
	  for (size_t ir = 0; ir < mc; ir += mr) {
	    for (size_t kr = 0; kr < kc; kr++) {
	      for (size_t i = ir; i < ir + mr; i++) {
		const auto Ac_p =
		  _mm256_broadcast_ss(Ac + i * kc + kr);

		for (size_t j = jr; j < jr + nr; j += SIMD) {
		  const auto Bc_p =
		    _mm256_load_ps(Bc + kr * nc + j);

		  auto Cc_p = _mm256_load_ps(Cc + i * nc + j);

		  _mm256_store_ps(
				  Cc + i * nc + j,
				  _mm256_fmadd_ps(Ac_p, Bc_p, Cc_p));
		}
	      }
	    }
	  }
	}

	for (size_t i = 0; i < mc; i++) {
	  for (size_t j = 0; j < nc; j += SIMD) {
	    const auto C_p =
	      _mm256_load_ps(C + (i + ic) * N + jc + j);
	    const auto Cc_p = _mm256_load_ps(Cc + i * nc + j);

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
  size_t N = 2048;

  std::default_random_engine engine;
  std::uniform_real_distribution<float> rnd;

  float *A = new float [N*N];
  float *B = new float [N*N];
  for (size_t i=0; i<N*N; i++) {
    A[i] = drand48();
    B[i] = drand48();
  }
  float *C_opt = nullptr;
  posix_memalign(reinterpret_cast<void **>(&C_opt), ALIGN,
		   N * N * sizeof(float));
  memset(C_opt, 0, N * N * sizeof(float));

  size_t flop = 2 * N * N * N;

  // Warmup
  //gemm_blocking(A, B, C_opt, N, N, N);
  double start = get_time();
  memset(C_opt, 0, N * N * sizeof(float));
  gemm_blocking(A, B, C_opt, N, N, N);
  double end = get_time();
  printf("%f GFLOPS.\n", flop / (end - start) / 1e9);

  float *C_naive = nullptr;
  posix_memalign(reinterpret_cast<void **>(&C_naive), ALIGN,
		   N * N * sizeof(float));
  memset(C_naive, 0, N * N * sizeof(float));
  gemm_naive(A, B, C_naive, N, N, N);

  check_error(C_naive, C_opt, N * N);

  free(A);
  free(B);
  free(C_naive);
  free(C_opt);

  return EXIT_SUCCESS;
}
