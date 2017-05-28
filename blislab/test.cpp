#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <limits.h>
#include <time.h>

typedef unsigned long long dim_t;

struct aux_s {
  float *b_next;
  float *b_next_s;
  char  *flag;
  int   pc;
  int   m;
  int   n;
};
typedef struct aux_s aux_t;

extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, float*,
                       int*, float*, int*, float*, float*, int*);

#define A( i, j ) A[ (j)*lda + (i) ]
#define B( i, j ) B[ (j)*ldb + (i) ]
#define C( i, j ) C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]

#define GEMM_SIMD_ALIGN_SIZE 32
#define SGEMM_MC 264
#define SGEMM_NC 128
#define SGEMM_KC 256
#define SGEMM_MR 24
#define SGEMM_NR 4
#define BL_MICRO_KERNEL bl_sgemm_asm_24x4

#include "bl_sgemm_asm_24x4.c"
#include "my_sgemm.c"

#define USE_SET_DIFF 1
#define TOLERANCE 1E0
void computeError(
        int    ldc,
        int    ldc_ref,
        int    m,
        int    n,
        float *C,
        float *C_ref
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if ( fabs( C( i, j ) - C_ref( i, j ) ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                break;
            }
        }
    }

}

void test_bl_sgemm(
        int m,
        int n,
        int k
        )
{
  float alpha = 1.0, beta = 1.0;
    int    i, j, p, nx;
    float *A, *B, *C, *C_ref;
    float tmp, error, flops;
    double ref_beg, ref_time, bl_sgemm_beg, bl_sgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_sgemm_rectime;

    A    = (float*)malloc( sizeof(float) * m * k );
    B    = (float*)malloc( sizeof(float) * k * n );

    lda = m;
    ldb = k;
    ldc = ( ( m - 1 ) / SGEMM_MR + 1 ) * SGEMM_MR;
    ldc_ref = m;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(float) );
    C_ref = (float*)malloc( sizeof(float) * m * n );

    nrepeats = 3;

    srand (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (float)( drand48() );
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (float)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (float)( 0.0 );
                C( i, j ) = (float)( 0.0 );
        }
    }

    for ( i = 0; i < nrepeats; i ++ ) {
        bl_sgemm_beg = omp_get_wtime();
        {
            bl_sgemm(
                    m,
                    n,
                    k,
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc
                    );
        }
        bl_sgemm_time = omp_get_wtime() - bl_sgemm_beg;

        if ( i == 0 ) {
            bl_sgemm_rectime = bl_sgemm_time;
        } else {
            bl_sgemm_rectime = bl_sgemm_time < bl_sgemm_rectime ? bl_sgemm_time : bl_sgemm_rectime;
        }
    }

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = omp_get_wtime();
        sgemm_( "N", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_ref, &ldc_ref );
        ref_time = omp_get_wtime() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

    computeError(
            ldc,
            ldc_ref,
            m,
            n,
            C,
            C_ref
            );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\n",
            m, n, k, flops / bl_sgemm_rectime, flops / ref_rectime );

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
}

int main( int argc, char *argv[] )
{
    int    m, n, k;

    if ( argc != 4 ) {
        printf( "Error: require 3 arguments, but only %d provided.\n", argc - 1 );
        exit( 0 );
    }

    sscanf( argv[ 1 ], "%d", &m );
    sscanf( argv[ 2 ], "%d", &n );
    sscanf( argv[ 3 ], "%d", &k );

    test_bl_sgemm( m, n, k );

    return 0;
}
