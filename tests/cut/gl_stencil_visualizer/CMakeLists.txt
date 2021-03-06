# 1) Change template_name to your test name
PROJECT (gl_stencil_visualizer)

SET (TEST_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)

# COPY ANY CONFIG FILES REQUIRED BY THE TEST
FILE(COPY test.conf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY L_host.mtx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# 3) Source files to compile
ADD_EXECUTABLE (${TEST_NAME} 
	main.cpp
	trackball.cpp
	visual_stencil.cpp
	stencil_visualizer.cpp
)

# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual arguments in quotes ("")) 
SET (ARGV
#	"--verbose"
# 	"--file=out.txt"
)


# REQUIRED packages
find_package(OpenGL)

find_package(Qt4) 
set(QT_USE_QTOPENGL TRUE)

INCLUDE(${QT_USE_FILE})


# Make sure the RBF.framework library is built and linked to this test
INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS} ${QT_INCLUDES})
ADD_DEPENDENCIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})

#TARGET_LINK_LIBRARIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${QT_LIBRARIES} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY} ${QT_QTCORE_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARY}) 
TARGET_LINK_LIBRARIES (${TEST_NAME} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${QT_LIBRARIES} ${OPENGL_LIBRARIES} )

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





