#include <random>
#include <vector>

#include <cassert>  // assert
#include <cstdio>   // sscanf, printf
#include <cstdlib>
#include <cstring>  // memset, memcpy

#include <sys/time.h>

#include <immintrin.h>

constexpr std::size_t ALIGN = 32;

constexpr std::size_t mc = 256;
constexpr std::size_t nc = 64;
constexpr std::size_t kc = 512;

constexpr std::size_t nr = 64;
constexpr std::size_t mr = 32;

constexpr std::size_t SIMD = (256 / sizeof(float) / 8);

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
                ::memcpy(Bc + p * nc, B + (p + pc) * N + jc,
                         nc * sizeof(float));
            }

            for (size_t ic = 0; ic < M; ic += mc) {
                alignas(ALIGN) float Ac[mc * kc];

                for (size_t i = 0; i < mc; i++) {
                    ::memcpy(Ac + i * kc, A + (i + ic) * K + pc,
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

void print_matrix(const float *matrix, const size_t M, const size_t N) {
    ::printf("%zu x %zu matrix\n", M, N);

    for (size_t m = 0; m < 32; m++) {
        for (size_t n = 0; n < 32; n++) {
            ::printf("%f ", matrix[m * N + n]);
        }
        ::putchar('\n');
    }
    ::putchar('\n');
}

// ./a.out M N K
int main(int argc, char *argv[]) {
    assert(argc == 4);

    std::size_t M, N, K;
    ::sscanf(argv[1], "%zu", &M);
    ::sscanf(argv[2], "%zu", &N);
    ::sscanf(argv[3], "%zu", &K);

    std::default_random_engine engine;
    std::uniform_real_distribution<float> rnd;

    float *A = nullptr;
    ::posix_memalign(reinterpret_cast<void **>(&A), ALIGN,
                     M * K * sizeof(float));
    for (size_t i = 0; i < M * K; i++) A[i] = rnd(engine);
    // ::print_matrix(A, M, K);

    float *B = nullptr;
    ::posix_memalign(reinterpret_cast<void **>(&B), ALIGN,
                     K * N * sizeof(float));
    for (size_t i = 0; i < K * N; i++) B[i] = rnd(engine);
    // ::print_matrix(B, K, N);

    float *C_opt = nullptr;
    ::posix_memalign(reinterpret_cast<void **>(&C_opt), ALIGN,
                     M * N * sizeof(float));
    ::memset(C_opt, 0, M * N * sizeof(float));

    size_t flop = 2 * M * N * K;

    // Warmup
    ::gemm_blocking(A, B, C_opt, M, N, K);

    ::memset(C_opt, 0, M * N * sizeof(float));

    double start = get_time();
    ::gemm_blocking(A, B, C_opt, M, N, K);
    double end = get_time();
    // ::print_matrix(C_opt, M, N);
    ::printf("%f GFLOPS.\n", flop / (end - start) / 1000 / 1000 / 1000);

    float *C_naive = nullptr;
    ::posix_memalign(reinterpret_cast<void **>(&C_naive), ALIGN,
                     M * N * sizeof(float));
    ::memset(C_naive, 0, M * N * sizeof(float));
    ::gemm_naive(A, B, C_naive, M, N, K);
    // ::print_matrix(C_naive, M, N);

    ::check_error(C_naive, C_opt, M * N);

    free(A);
    free(B);
    free(C_naive);
    free(C_opt);

    return EXIT_SUCCESS;
}
