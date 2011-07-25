###############################################
# Build Options (Definitions and compiler flags)
###############################################
	# Used by ALL compilers
	#ADD_DEFINITIONS("-g -shared-intel -openmp")
	ADD_DEFINITIONS(-g)
#SET (CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS_INIT} "-openmp" CACHE STRING "Flags used by the linker during the creation of dll's.")
#SET (CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS_INIT} "-openmp" CACHE STRING "Flags used by the linker during the creation of dll's.")
#SET (CMAKE_MODULE_LINKER_FLAGS ${CMAKE_MODULE_LINKER_FLAGS_INIT} "-openmp" CACHE STRING "Flags used by the linker during the creation of dll's.")

	# Used by SPECIFIC compilers
 	# SET (CMAKE_CXX_FLAGS)


###############################################
# EXTENSIONS TO INCLUDE: 
###############################################
	ENABLE_TESTING()
	INCLUDE (CPack)
	INCLUDE (UseDoxygen)
	FIND_PACKAGE (MPI)

OPTION (USE_VTK "Enable/Disable the use of VTK (if required by a test)" ON)
OPTION (USE_CUDA "Enable/Disable the use of CUDA" ON)
OPTION (USE_OPENCL "Enable/Disable the use of OPENCL" ON)

	FIND_PACKAGE (OPENCL)
    IF (NOT USE_OPENCL)
        set (OPENCL_FOUND false)
    ENDIF (NOT USE_OPENCL)


	unset(CUDA_CUDA_LIBRARY CACHE)
	FIND_PACKAGE (CUDA)
	IF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})
		MESSAGE (WARNING "\nWARNING! Dep: libcuda was NOT FOUND. Disabling cuda support for framework. Please install NVidia proprietary driver and verify you have an NVidia GPU if you want it enabled.\n")
		set(CUDA_FOUND false)
	ENDIF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})
    IF (NOT USE_CUDA)
        set (CUDA_FOUND false)
    ENDIF (NOT USE_CUDA)


###############################################
# Locate Required Libraries
###############################################
	# Find library: find_library(<VAR> name1 [path1 path2 ...])

	# Download and install Armadillo separately. 
	# Specify local installation dir here. If installed globally the dir is unnecessary.
	FIND_LIBRARY (armadillo armadillo PATHS
	# efb06: bollig account
	/Users/erlebach/Documents/src/armadillo-0.9.52 
	~/local/usr/lib64
	~/local/usr/lib
	~/local/lib 
	~/local/lib64
	/usr/lib
	/usr/local/lib
	/usr/lib64
	/usr/local/lib64
	NO_DEFAULT_PATH
	)
	MESSAGE(STATUS "Found armadillo in: ${armadillo}")

    SET(CMAKE_FIND_LIBRARY_SUFFIXES_SAVED ${CMAKE_FIND_LIBRARY_SUFFIXES}) #Backup
    LIST(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.3")

	# Typically installed separately. Same rules as Armadillo (local dir here; global unecessary)
   	FIND_LIBRARY(fftw3 fftw3 PATHS 
		/Users/erlebach/Documents/src/fftw-3.2.2/.libs
	~/local/usr/lib64
	~/local/usr/lib
	~/local/lib 
	~/local/lib64
	/usr/lib
	/usr/local/lib
	/usr/lib64
	/usr/local/lib64
    #	NO_DEFAULT_PATH
	)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVED}) #Restore
	MESSAGE(STATUS "Found fftw3 in: ${fftw3}")



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
        $ENV{INCLUDE_PATH}
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
