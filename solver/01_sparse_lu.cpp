#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "mkl_pardiso.h"
#include "mkl_types.h"

int main(int argc, char **argv) {
  int n = 8;
  int ia[9] = { 1, 5, 8, 10, 12, 15, 17, 18, 19};
  int ja[18] = 
    { 1,    3,       6, 7,
         2, 3,    5,
            3,             8,
               4,       7,
                  5, 6, 7,
                     6,    8,
                        7,
                           8
    };
  double a[18] = 
    { 7.0,      1.0,           2.0, 7.0,
          -4.0, 8.0,      2.0,
                1.0,                     5.0,
                     7.0,           9.0,
                          5.0, 1.0, 5.0,
                              -1.0,      5.0,
                                   11.0,
                                         5.0
    };
  int mtype = -2;       /* Real symmetric matrix */
  double b[8], x[8];    /* RHS and solution vectors. */
  int nrhs = 1;         /* Number of right hand sides. */
  void *pt[64];         /* Internal solver memory pointer pt, */
  int iparm[64];        /* Pardiso control parameters. */
  double ddum;          /* Double dummy */
  int idum;             /* Integer dummy. */
  for (int i=0; i<64; i++) iparm[i] = 0;
  iparm[0] = 1;         /* No solver default */
  iparm[1] = 2;         /* Fill-in reordering from METIS */
  iparm[3] = 0;         /* No iterative-direct algorithm */
  iparm[4] = 0;         /* No user fill-in reducing permutation */
  iparm[5] = 0;         /* Write solution into x */
  iparm[6] = 0;         /* Not in use */
  iparm[7] = 2;         /* Max numbers of iterative refinement steps */
  iparm[8] = 0;         /* Not in use */
  iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
  iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
  iparm[11] = 0;        /* Not in use */
  iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off */
  iparm[13] = 0;        /* Output: Number of perturbed pivots */
  iparm[14] = 0;        /* Not in use */
  iparm[15] = 0;        /* Not in use */
  iparm[16] = 0;        /* Not in use */
  iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1;       /* Output: Mflops for LU factorization */
  iparm[19] = 0;        /* Output: Numbers of CG Iterations */
  int maxfct = 1;       /* Maximum number of numerical factorizations. */
  int mnum = 1;         /* Which factorization to use. */
  int msglvl = 1;       /* Print statistical information in file */
  int error = 0;        /* Initialize error flag */
  for (int i=0; i<64; i++) pt[i] = 0;
  int phase = 11;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  phase = 22;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  phase = 33;
  for (int i=0; i<n; i++) b[i] = 1;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
  printf ("\nThe solution of the system is: ");
  for (int i=0; i<n; i++) {
    printf ("\n x [%d] = % f", i, x[i]);
  }
  printf ("\n");
  phase = -1; /* Release internal memory. */
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  return 0;
}
