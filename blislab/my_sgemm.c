float *bl_malloc_aligned(int m, int n, int size) {
  float *ptr;
  int    err;
  err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );
  if (err) {
    printf( "bl_malloc_aligned(): posix_memalign() failures" );
    exit(1);
  }
  return ptr;
}

inline void packA_mcxkc_d(int m, int k, float *XA, int ldXA, int offseta, float *packA) {
  float *a_pntr[SGEMM_MR];
  for (int i=0; i<m; i++) {
    a_pntr[i] = XA + offseta + i;
  }
  for (int i=m; i<SGEMM_MR; i++) {
    a_pntr[i] = XA + offseta + 0;
  }
  for (int p=0; p<k; p++) {
    for (int i=0; i<SGEMM_MR; i++) {
      *packA = *a_pntr[i];
      packA++;
      a_pntr[i] = a_pntr[i] + ldXA;
    }
  }
}

inline void packB_kcxnc_d(int n, int k, float *XB, int ldXB, int offsetb, float *packB) {
  float *b_pntr[SGEMM_NR];
  for (int j=0; j<n; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + j);
  }
  for (int j=n; j<SGEMM_NR; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + 0);
  }

  for (int p=0; p<k; p++) {
    for (int j=0; j<SGEMM_NR; j++) {
      *packB++ = *b_pntr[j]++;
    }
  }
}

void bl_macro_kernel(int m, int n, int k, float *packA, float *packB, float *C, int ldc) {
  aux_t aux;
  aux.b_next = packB;
  for (int j=0; j<n; j+=SGEMM_NR ) {
    aux.n = std::min(n-j, SGEMM_NR);
    for (int i=0; i<m; i+=SGEMM_MR) {
      aux.m = std::min(m-i, SGEMM_MR);
      if (i+SGEMM_MR >= m) {
        aux.b_next += SGEMM_NR * k;
      }
      micro_kernel(k, &packA[i*k], &packB[j*k], &C[j*ldc+i], ldc, &aux);
    }
  }
}

// C must be aligned
void bl_sgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        )
{
    int    i, j, p, bl_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    float *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_sgemm(): early return\n" );
        return;
    }

    // sequential is the default situation
    bl_ic_nt = 1;
    // check the environment variable
    str = getenv( "BLISLAB_IC_NT" );
    if ( str != NULL ) {
        bl_ic_nt = (int)strtol( str, NULL, 10 );
    }

    // Allocate packing buffers
    packA  = bl_malloc_aligned( SGEMM_KC, ( SGEMM_MC + 1 ) * bl_ic_nt, sizeof(float) );
    packB  = bl_malloc_aligned( SGEMM_KC, ( SGEMM_NC + 1 )            , sizeof(float) );

    for ( jc = 0; jc < n; jc += SGEMM_NC ) {                       // 5-th loop around micro-kernel
      jb = std::min( n - jc, SGEMM_NC );
        for ( pc = 0; pc < k; pc += SGEMM_KC ) {                   // 4-th loop around micro-kernel
          pb = std::min( k - pc, SGEMM_KC );

            #pragma omp parallel for num_threads( bl_ic_nt ) private( jr )
            for ( j = 0; j < jb; j += SGEMM_NR ) {
                packB_kcxnc_d(
          std::min( jb - j, SGEMM_NR ),
                        pb,
                        &XB[ pc ],
                        k, // should be ldXB instead
                        jc + j,
                        &packB[ j * pb ]
                        );
            }

            #pragma omp parallel for num_threads( bl_ic_nt ) private( ic, ib, i, ir )
            for ( ic = 0; ic < m; ic += SGEMM_MC ) {              // 3-rd loop around micro-kernel
                int     tid = omp_get_thread_num();
                ib = std::min( m - ic, SGEMM_MC );

                for ( i = 0; i < ib; i += SGEMM_MR ) {
                    packA_mcxkc_d(
                                  std::min( ib - i, SGEMM_MR ),
                            pb,
                            &XA[ pc * lda ],
                            m,
                            ic + i,
                            &packA[ tid * SGEMM_MC * pb + i * pb ]
                            );
                }

                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA  + tid * SGEMM_MC * pb,
                        packB,
                        &C[ jc * ldc + ic ],
                        ldc
                        );

            }                                                  // End 3.rd loop around micro-kernel
        }                                                      // End 4.th loop around micro-kernel
    }                                                          // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}
