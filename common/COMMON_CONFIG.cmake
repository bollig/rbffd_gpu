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

	unset(CUDA_CUDA_LIBRARY CACHE)
	FIND_PACKAGE (CUDA)
	IF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})
		MESSAGE (WARNING "\nWARNING! Dep: libcuda was NOT FOUND. Disabling cuda support for framework. Please install NVidia proprietary driver and verify you have an NVidia GPU if you want it enabled.\n")
		set(CUDA_FOUND false)
	ENDIF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})


###############################################
# Locate Required Libraries
###############################################
	# Find library: find_library(<VAR> name1 [path1 path2 ...])

	# Download and install Armadillo separately. 
	# Specify local installation dir here. If installed globally the dir is unnecessary.
	FIND_LIBRARY (armadillo armadillo PATHS
	# efb06: bollig account
	/Users/erlebach/Documents/src/armadillo-0.9.52 
	/usr/local/lib64 
	~/local/usr/lib64
	~/local/usr/lib
	~/local/lib 
	~/local/lib64)

	# Typically installed separately. Same rules as Armadillo (local dir here; global unecessary)
   	FIND_LIBRARY(fftw3 fftw3 PATHS /Users/erlebach/Documents/src/fftw-3.2.2/.libs)



###############################################
# Setup GPU APIs
###############################################





###############################################
# External dependency search paths
###############################################
	# Directories searched for headers ORDER does not matter. 
	# If a directory does not exist it is skipped
	SET (RBF_INCLUDE_DIRS 
		.
		/Users/erlebach/Documents/src/fftw-3.2.2/include
		/Users/erlebach/Documents/src/armadillo-0.9.52/include
		/opt/local/include   # for boost
                ~/local/include
		~/local/usr/include
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
