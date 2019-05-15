include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

02_petsc: 02_petsc.o chkopts
	-${CLINKER} 02_petsc.o ${PETSC_KSP_LIB}
	${RM} 02_petsc.o

include ${PETSC_DIR}/lib/petsc/conf/test
