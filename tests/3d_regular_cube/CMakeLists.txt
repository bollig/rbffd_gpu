#cmake_minimum_required(VERSION 2.6)


# 1) Change template_name to your test name
PROJECT (3d_regular_cube)

SET (TEST_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)

# 3) Source files to compile
ADD_EXECUTABLE (${TEST_NAME} 
	main.cpp
)

# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual arguments in quotes ("")) 
SET (ARGV
#	"--verbose"
# 	"--file=out.txt"
)


# Make sure the RBF.framework library is built and linked to this test
INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
ADD_DEPENDENCIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})
TARGET_LINK_LIBRARIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES}) 

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





