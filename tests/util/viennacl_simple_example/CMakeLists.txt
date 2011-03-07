# 1) Change template_name to your test name
PROJECT (viennacl_simple_example)

SET (TEST_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)


# THIS TEST CAN ONLY WORK WITH CUDA PRESENT
IF (OPENCL_FOUND) 

	# Make sure the RBF.framework library is built and linked to this test
	INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})

	# 3) Source files to compile
	ADD_EXECUTABLE (${TEST_NAME} 
		main.cpp
	)

	# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual arguments in quotes ("")) 
	SET (ARGV
        #	"--verbose"
        # 	"--file=out.txt"
        )

        ADD_DEPENDENCIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})
        TARGET_LINK_LIBRARIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${OPENCL_LIBRARIES})

        # 5) Add a parallel or serial test
        #IF (MPI_FOUND AND USE_MPI)
        #	TARGET_LINK_LIBRARIES (${TEST_NAME}
        #		${MPI_LIBRARIES}
        #	)
        #
                # Add a Parallel Test
                # Format: ADD_TEST( [TestName] [MPIExecutable] [MPINumProcFlag] [#ofProcs] [MPIOptions] [Executable] [Arg1] [Arg2] ... [ArgN])
                # ADD_TEST (${TEST_NAME}_test1 ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS} ${MPIEXEC_PREFLAGS} "${TEST_NAME}" ${ARGV} ${MPIEXEC_POSTFLAGS})
        #ELSE ()
                # Add a Serial Test:
                # Format: ADD_TEST( [TestName] [Executable] [Arg1] [Arg2] ... [ArgN])
        #	ADD_TEST (${TEST_NAME}_test1 ${TEST_NAME} ${ARGV})
        #ENDIF (MPI_FOUND AND USE_MPI)

        ADD_TEST (${TEST_NAME}_test1 "${TEST_NAME}" ${ARGV})
ELSE(OPENCL_FOUND) 
    MESSAGE (WARNING "WARNING! Test: ${TEST_NAME} will not be built because OPENCL was not found. Please install CUDA v3.1 on your system!")
ENDIF(OPENCL_FOUND) 

