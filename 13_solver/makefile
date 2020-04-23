include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

00_lapack:
	icpc -mkl=parallel 00_lapack.cpp -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

01_pardiso:
	icpc -mkl=parallel 01_pardiso.cpp -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

02_petsc: 02_petsc.o chkopts
	-${CLINKER} 02_petsc.o ${PETSC_KSP_LIB}
	${RM} 02_petsc.o

include ${PETSC_DIR}/lib/petsc/conf/test
