#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <immintrin.h>

int main(int argc, char **argv) {
  //int NALIGN = 32;
  int N = atoi(argv[1]);
  float * A = new float  [N*N];
  float * B = new float  [N*N];
  float * C = new float  [N*N];
  for (int i=0; i<N*N; i++) {
    A[i] = drand48();
    B[i] = drand48();
    C[i] = 0;
  }

//first touch
  const int m = N, n = N, k = N;
  const int kc = 512;
  const int nc = 64;
  const int mc = 64;
  const int nr = 32;
  const int mr = 32;
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          B[(p+pc)*N+j+jc] = drand48();
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            A[(i+ic)*N+p+pc] = drand48();
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[(i+ic)*N+j+jc] = 0;
          }
        }
      }
    }
  }
//first touch end

  struct timeval tic, toc;
  gettimeofday(&tic, NULL);


  float Ac[mc*kc];
  float Bc[kc*nc];
  float Cc[mc*nc];
#pragma omp parallel for private(Ac,Bc,Cc)
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          //__m256 B_tmp = _mm256_load_ps(B+(p+pc)*N+j+jc);
          //_mm256_store_ps(Bc+p*nc+j,B_tmp);
          Bc[p*nc+j] = B[(p+pc)*N+j+jc];
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = A[(i+ic)*N+p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                //avx&fmadd
                __m256 a_tmp = _mm256_broadcast_ss(Ac+(i*kc+kr));
                __m256 c_tmp = _mm256_load_ps(Cc+(i*nc+jr));
                __m256 b_tmp = _mm256_load_ps(Bc+(kr*nc+jr));
                __m256 c_tmp2 = _mm256_load_ps(Cc+(i*nc+jr+8));
                __m256 b_tmp2 = _mm256_load_ps(Bc+(kr*nc+jr+8));
                __m256 c_tmp3 = _mm256_load_ps(Cc+(i*nc+jr+16));
                __m256 b_tmp3 = _mm256_load_ps(Bc+(kr*nc+jr+16));
                __m256 c_tmp4 = _mm256_load_ps(Cc+(i*nc+jr+24));
                __m256 b_tmp4 = _mm256_load_ps(Bc+(kr*nc+jr+24));
                //__m256 out_tmp = _mm256_mul_ps(a_tmp,b_tmp);
                //__m256 out = _mm256_add_ps(c_tmp,out_tmp);
                __m256 out = _mm256_fmadd_ps(a_tmp,b_tmp,c_tmp);
                __m256 out2 = _mm256_fmadd_ps(a_tmp,b_tmp2,c_tmp2);
                __m256 out3 = _mm256_fmadd_ps(a_tmp,b_tmp3,c_tmp3);
                __m256 out4 = _mm256_fmadd_ps(a_tmp,b_tmp4,c_tmp4);
                _mm256_store_ps(Cc+(i*nc+jr),out);
                _mm256_store_ps(Cc+(i*nc+jr+8),out2);
                _mm256_store_ps(Cc+(i*nc+jr+16),out3);
                _mm256_store_ps(Cc+(i*nc+jr+24),out4);
                //Cc[i*nc+j] += Ac[i*kc+kr] * Bc[kr*nc+j];
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            //__m256 C_tmp = _mm256_load_ps(C+(i+ic)*N+j+jc);
            //__m256 Cc_tmp = _mm256_load_ps(Cc+i*nc+j);
            //C_tmp = _mm256_add_ps(C_tmp,Cc_tmp);
            //_mm256_store_ps(C+(i+ic)*N+j+jc,C_tmp);
            C[(i+ic)*N+j+jc] += Cc[i*nc+j];
          }
        }
      }
    }
  }
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C[i*N+j] -= A[i*N+k] * B[k*N+j];
      }
    }
  }
  float err = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      err += fabs(C[i*N+j]);
    }
  }
  printf("error: %f\n",err/N/N);
  delete[] A;
  delete[] B;
  delete[] C;
}
