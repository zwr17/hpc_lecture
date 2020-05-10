#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>
#include <immintrin.h>

#include <mpi.h>
#include <mkl.h>


void matmal_simd_bs(double *A, double *B, double *C, int N, int M, int K);	
	
double time_diff(struct timeval st, struct timeval et){
	return et.tv_sec-st.tv_sec+(et.tv_usec-st.tv_usec)*1e-6;
}

// 行列A,Cの横分割
void divide(int N, int mpirank, int mpisize, int *st, int *et){
	int len = (N + mpisize - 1)/mpisize;
	int s = len*mpirank;
	int t = len*(mpirank+1);
	
	if(s > N) s = N;
	if(t > N) t = N;

	*st = s;
	*et = t;

	return;
}

int main(int args, char *argv[]){
	int mpirank, mpisize;
	MPI_Init(&args, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	
	//printf("mpisize = %d, mpirank = %d\n", mpisize, mpirank);
	
	struct timeval st, et;
	int N = atoi(argv[1]); // 行列サイズ
	double *A  = (double *)malloc(sizeof(double) * N * N);
	double *B  = (double *)malloc(sizeof(double) * N * N);
	double *C1 = (double *)malloc(sizeof(double) * N * N);
	double *C2 = (double *)malloc(sizeof(double) * N * N);
	double time;

	// check
	if(A==NULL || B==NULL || C1==NULL || C2==NULL){
		printf("Malloc failed\n");
		return -1;
	}

	// init
	if( mpirank == 0 ){
		for(int i=0; i<N; i++){
			for(int j=0; j<N; j++){
				A[i*N + j] = drand48();
				B[i*N + j] = drand48();
			}
		}
	}

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			C1[i*N + j] = 0.0;
			C2[i*N + j] = 0.0;
		}
	}
	
	// 正しい値の計算
	if(mpirank == 0){
		double alpha = 1.0, beta = 0.0;
		gettimeofday(&st, NULL);
		cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C1, N);
		gettimeofday(&et, NULL);
		
		time = time_diff(st, et);

		printf("rank=%d, N=%d: %lf s (%lf GFlops)\n", mpirank, N, time, 2.0*N*N*N/time/1e9);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	// MPIを使用した並列化
	int sp, ep;

	// send, recv, for communication
	int *counts = (int *)malloc(sizeof(int) * mpisize);
	int *displs = (int *)malloc(sizeof(int) * mpisize);
	
	for(int j=0; j<mpisize; j++){
		divide(N, j, mpisize, &sp, &ep);
		counts[j] = (ep - sp)*N;
	}
	
	displs[0] = 0;
	for(int j=1; j<mpisize; j++){
		displs[j] = displs[j-1] + counts[j];
	}
	
	// each proces
	divide(N, mpirank, mpisize, &sp, &ep);
	// 分割の確認
	//printf("mpirank = %d, sp = %d, ep = %d\n", mpirank, sp, ep);
	int length = ep-sp;
	
	double *LA = (double *)malloc(sizeof(double) * N * length);
	double *LC = (double *)malloc(sizeof(double) * N * (N + mpisize - 1)/mpisize);
	
	if(LA == NULL || LC == NULL){
		printf("Malloc failed\n");
		return -1;
	}
	
	// init
	for(int i=0; i<length; i++){
		for(int j=0; j<N; j++){
			LC[i*N + j] = 0.0;
		}
	}

	// 計算本体　データ転送　→　計算　→　データ転送　を計測
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&st, NULL);
	
	MPI_Scatterv(A, counts, displs, MPI_DOUBLE, LA, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	matmal_simd_bs(LA, B, LC, N, length, N);
	MPI_Gatherv(LC, N*length, MPI_DOUBLE, C2, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&et, NULL);
	
	// error フロベニウスノルム
	double alpha = -1.0;
	const int incn = 1;
	cblas_daxpy( N, alpha, C1, incn, C2, incn);
	double error = cblas_dnrm2( N, C2, incn);

	if(mpirank == 0){
		time = time_diff(st, et);
		printf("rank=%d, N=%d: %lf s (%lf GFlops)\n", mpirank, N, time, 2.0*N*N*N/time/1e9);
		printf("error = %f \n", error);	
	}

	MPI_Finalize();
	return 0;
}

// 分割での行列計算
void matmal_simd_bs(double *A, double *B, double *C, int N, int M, int K){	
	const int n = N;
	const int m = M;
	const int k = K;
	
	const int kc = 512;
	const int nc = 32;
	const int mc = 256;
	const int nr = 32;
	const int mr = 32;	
	
	double Ac[mc*kc];
	double Bc[kc*nc];
	double Cc[mc*nc];
	
	int tp1, tp2;

	#pragma omp parallel for private(Ac, Bc, Cc, tp1, tp2)
	for(int jc=0; jc<n; jc+=nc){
		for(int pc=0; pc<k; pc+=kc){
			for(int p=0; p<(int)std::fmin(kc, k-pc); p++){
				tp1 = p*nc;
				tp2 = (p + pc)*n;
				for(int j=0; j<(int)std::fmin(nc, n-jc); j++){
					Bc[tp1 + j] = B[tp2 + j + jc];
				}
				for(int j=(int)std::fmin(nc, n-jc); j<nc; j++){
					Bc[tp1 + j] = 0.0;
				}
			}
			for(int p=(int)std::fmin(kc, k-pc); p<kc; p++){
				tp1 = p*nc;
				for(int j=0; j<nc; j++){
					Bc[tp1 + j] = 0.0;
				}
			}
			for(int ic=0; ic<m; ic+=mc){
				for(int i=0; i<(int)std::fmin(mc, m-ic); i++){
					tp1 = i*kc;
					tp2 = (i + ic)*k;
					for(int p=0; p<(int)std::fmin(kc, k - pc); p++){
						Ac[tp1 + p] = A[tp2 + p + pc];
					}
					for(int p=(int)std::fmin(kc, k - pc); p<kc; p++){
						Ac[tp1 + p] = 0.0;
					}
					tp1 = i*nc;
					for(int j=0; j<nc; j++){
						Cc[tp1 + j] = 0.0;
					}
				}
				for(int i=(int)std::fmin(mc, m-ic); i<mc; i++){
					tp1 = i*kc;
					for(int p=0; p<kc; p++){
						Ac[tp1 + p] = 0.0;
					}
					tp1 = i*nc;
					for(int j=0; j<nc; j++){
						Cc[tp1 + j] = 0.0;
					}
				}

				for(int jr=0; jr<nc; jr+=nr){
					for(int ir=0; ir<mc; ir+=mr){
						
						for(int i=ir; i<ir+mr; i++){
							for(int kr=0; kr<kc; kr++){
								
								__m256d va = _mm256_broadcast_sd(Ac + i*kc + kr);
								
								for(int j=jr; j<jr+nr; j+=4){
									__m256d vb = _mm256_load_pd(Bc + kr*nc + j);
									__m256d vc = _mm256_load_pd(Cc + i*nc + j);
									__m256d vd = _mm256_mul_pd(va, vb);
									vc = _mm256_add_pd(vc, vd);
									_mm256_store_pd(Cc + i*nc + j, vc);
								}
							}
						}

					}
				}	
				for(int i=0; i<(int)std::fmin(mc, m-ic); i++){
					tp1 = (i + ic)*n;
					tp2 = i * nc;
					for(int j=0; j<nc; j++){
						C[tp1 + j + jc] += Cc[tp2 + j];
					}
				}	
			}
		}
	}

	return;
}


