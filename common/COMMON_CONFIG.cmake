###############################################
# Build Options (Definitions and compiler flags)
###############################################
	# Used by ALL compilers
	ADD_DEFINITIONS(-g)
	# Used by SPECIFIC compilers
 	# SET (CMAKE_CXX_FLAGS)


###############################################
# EXTENSIONS TO INCLUDE: 
###############################################
	ENABLE_TESTING()
	INCLUDE (CPack)
	INCLUDE (UseDoxygen)
	FIND_PACKAGE (MPI)
	FIND_PACKAGE (OPENCL)
	FIND_PACKAGE (CUDA)


###############################################
# Locate Required Libraries
###############################################
	# Find library: find_library(<VAR> name1 [path1 path2 ...])

	# Download and install Armadillo separately. 
	# Specify local installation dir here. If installed globally the dir is unnecessary.
	FIND_LIBRARY (armadillo armadillo PATHS
	# efb06: bollig account
	/Users/erlebach/Documents/src/armadillo-0.6.12 
	/usr/local/lib64 
	~/local/lib 
	~/local/lib64)

	# Typically installed separately. Same rules as Armadillo (local dir here; global unecessary)
	FIND_LIBRARY(fftw fftw3 PATHS /Users/erlebach/Documents/src/fftw-3.2.2/.libs)



###############################################
# Setup GPU APIs
###############################################

IF (CUDA_FOUND) 
	MESSAGE(STATUS "FOUND CUDA, NOT ENABLED YET")
ENDIF (CUDA_FOUND)

IF (OPENCL_FOUND) 
	MESSAGE(STATUS "FOUND OPENCL, NOT ENABLED YET")
ENDIF (OPENCL_FOUND)



###############################################
# External dependency search paths
###############################################
	# Directories searched for headers ORDER does not matter. 
	# If a directory does not exist it is skipped
	SET (RBF_INCLUDE_DIRS 
		.
		/Users/erlebach/Documents/src/fftw-3.2.2/include
		/Users/erlebach/Documents/src/armadillo-0.6.12/include
		/opt/local/include   # for boost
                ~efb06/local/include
	)
		
	INCLUDE_DIRECTORIES ( ${RBF_INCLUDE_DIRS} )

###############################################
# Setup MPI 
###############################################

OPTION (USE_MPI "Enable/Disable parallel build and linking with MPI" ON)
IF (MPI_FOUND AND USE_MPI)
	SET (CMAKE_CXX_FLAGS ${MPI_COMPILE_FLAGS})
	SET (CMAKE_C_FLAGS ${MPI_COMPILE_FLAGS})
	SET (CMAKE_LINK_FLAGS ${MPI_LINK_FLAGS})
	
	INCLUDE_DIRECTORIES (${MPI_INCLUDE_PATH})
	# NOTE: add a target_link_library( MPI_LIBRARIES) for libs and bins
	# TESTS that run parallel should use MPIEXEC
ENDIF (MPI_FOUND AND USE_MPI) 
