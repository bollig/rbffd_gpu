IF (USE_GCC) 
    set (CMAKE_C_COMPILER "gcc")
    set (CMAKE_CXX_COMPILER "g++")      
ENDIF (USE_GCC)

IF (USE_GCC46) 
    set (CMAKE_C_COMPILER "gcc-4.6")
    set (CMAKE_CXX_COMPILER "g++-4.6")      
ENDIF (USE_GCC46)

IF (USE_APPLE_GCC) 
    set (CMAKE_C_COMPILER "llvm-gcc")
    set (CMAKE_CXX_COMPILER "llvm-g++")      
ENDIF (USE_APPLE_GCC)



IF (USE_GCC47) 
    set (CMAKE_C_COMPILER "gcc-4.7")
    set (CMAKE_CXX_COMPILER "g++-4.7")      
ENDIF (USE_GCC47)


IF (USE_GCC44) 
    set (CMAKE_C_COMPILER "gcc-4.4")
    set (CMAKE_CXX_COMPILER "g++-4.4")      
ENDIF (USE_GCC44)


IF (USE_GCC44_KEENELAND) 
    set (CMAKE_C_COMPILER "gcc44")
    set (CMAKE_CXX_COMPILER "g++44")      
ENDIF (USE_GCC44_KEENELAND)


IF (USE_ICC)
    set (CMAKE_C_COMPILER "icc")
    set (CMAKE_CXX_COMPILER "icpc")
ENDIF (USE_ICC) 




PROJECT(rbffd_gpu)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
	

OPTION (USE_VTK "Enable/Disable the use of VTK (if required by a test)" ON)
OPTION (USE_CUDA "Enable/Disable the use of CUDA" ON)
OPTION (USE_OPENCL "Enable/Disable the use of OPENCL" ON)
OPTION (USE_ICC "Enable/Disable the use of Intel ICC compiler for CUDA" OFF)
OPTION (USE_MPI "Enable/Disable parallel build and linking with MPI" ON)

MESSAGE (STATUS "USE_VTK = ${USE_VTK}")
MESSAGE (STATUS "USE_CUDA = ${USE_CUDA}")
MESSAGE (STATUS "USE_OPENCL = ${USE_OPENCL}")
MESSAGE (STATUS "USE_ICC = ${USE_ICC}")
MESSAGE (STATUS "USE_MPI = ${USE_MPI}")



SET( CMAKE_MODULE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/common
	${CMAKE_CURRENT_SOURCE_DIR}/common/UseDoxygen
	${CMAKE_CURRENT_SOURCE_DIR}/common/external/jedbrown-cmake-modules
	${CMAKE_CURRENT_SOURCE_DIR}/common/external/armadillo
)

# Setup any external libraries we might need
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/external)
# Get common build instructions used by all operating systems
INCLUDE (COMMON_CONFIG)

################ BEGIN OS DEPENDENT CONFIGS ###################

###############################################
# 		APPLE OSX 10.6
###############################################
IF(APPLE)
    INCLUDE (OSX_CONFIG)
ENDIF(APPLE)

###############################################
# 		UBUNTU LINUX 9.10
###############################################
IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    INCLUDE (LINUX_CONFIG)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

###############################################
# 		WINDOWS 7
###############################################
IF (WIN32)
    INCLUDE (WIN_CONFIG)
ENDIF(WIN32)

################# END OS DEPENDENT CONFIGS ####################

# These must come AFTER the external subdir so we have 
# access to CMake variables for each dependency
SET (FRAMEWORK_LIBRARY RBF)	# Name of the framework lib (i.e., lib<name>.dylib)
SET(FRAMEWORK_CUDA_LIBRARY ${FRAMEWORK_LIBRARY}_cuda)
SET(FRAMEWORK_OPENCL_LIBRARY ${FRAMEWORK_LIBRARY}_opencl)
SET (FRAMEWORK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

STRING(REPLACE ":" ";" inc_paths "$ENV{INCLUDE_PATH}:.")
include_directories(/opt/hpc/intel/mvapich/include/ ${inc_paths})

# This is an external dependency
# Its name should not change
set(MPI_LIB $ENV{I_MPI_ROOT}/lib64)  # module -intel-impi-x86_64
message("*** MPI_LIB = ${MPI_LIB}")
SET( FRAMEWORK_DEPENDENCIES
	$ENV{EXTERNAL_LIB_DEPENDENCIES}
	${Boost_LIBRARIES}
        #$ENV{I_MPI_ROOT}/intel64/lib/libmpi.so   # on hpc (GE)
        ${MPI_LIB}/libmpich.so   # on hpc (GE) Not sure what this is. 
        #/opt/hpc/gnu/openmpi/lib64/libmpi.so
	#${CUDA_LIBRARIES}
)

message("--------------")
message("${Boost_LIBRARIES}")

# This is a neat trick: since the external deps are now part of
# this source tree we can reference their subtree as ${<name>_SOURCE_DIR}
# so long as CMake is told that they are subdirs (i.e., add_subdirectory) 
message("external include dirs **** : ${EXTERNAL_INCLUDE_DIRS}")
message("mpi include dirs **** : ${MPI_INCLUDE}")
SET( FRAMEWORK_DEP_INCLUDE_DIRS
	$ENV{EXTERNAL_INCLUDE_DIRS}
	${MPI_INCLUDE}  # on hpc (GE)
	#/opt/hpc/gnu/openmpi/include/    # added by GE
	#${timing_SOURCE_DIR}
	#${CUDA_INCLUDE_DIRS}
)

#add_definitions( -DUSE_DOUBLE_VEC3 -DUSE_KDTREE -O2 -no-vec -opt-report -vec-report -par-report)
#ICC:  add_definitions( -DUSE_DOUBLE_VEC3 -DUSE_KDTREE -O2 -opt-report -vec-report -par-report)
add_definitions(-DUSE_DOUBLE_VEC3 -DUSE_KDTREE -Wall 
  #-std=c++0x 
    -O2 
    #-g 
    -DNDEBUG 
    #-DTHRUST_DEBUG )  
    ) 

message(STATUS "CompilerID: ${CMAKE_CXX_COMPILER_ID}")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(
      -DUSE_CLANG
      -DDEBUG_ALL
	 # -g 
      -stdlib=libc++
  )
endif()

#INCLUDE_DIRECTORIES($ENV{EXTERNAL_INCLUDE_DIRS})

#GET_DIRECTORY_PROPERTY(incdirs INCLUDE_DIRECTORIES) 
#MESSAGE ("INCLUDE_DIRECTORIES: ${incdirs}")

# Make sure we build library and tests subtrees
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/src)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/tests)

