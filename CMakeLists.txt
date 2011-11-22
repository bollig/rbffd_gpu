IF (USE_GCC) 
    set (CMAKE_C_COMPILER "gcc")
    set (CMAKE_CXX_COMPILER "g++")      
ENDIF (USE_GCC)

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




PROJECT(RBF_framework)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
	
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
include_directories(${inc_paths})

# This is an external dependency
# Its name should not change
SET( FRAMEWORK_DEPENDENCIES
	$ENV{EXTERNAL_LIB_DEPENDENCIES}
	#${CUDA_LIBRARIES}
)

# This is a neat trick: since the external deps are now part of
# this source tree we can reference their subtree as ${<name>_SOURCE_DIR}
# so long as CMake is told that they are subdirs (i.e., add_subdirectory) 
SET( FRAMEWORK_DEP_INCLUDE_DIRS
	$ENV{EXTERNAL_INCLUDE_DIRS}
	#${timing_SOURCE_DIR}
	#${CUDA_INCLUDE_DIRS}
)

#add_definitions( -DUSE_DOUBLE_VEC3 -DUSE_KDTREE -O2 -no-vec -opt-report -vec-report -par-report)
#ICC:  add_definitions( -DUSE_DOUBLE_VEC3 -DUSE_KDTREE -O2 -opt-report -vec-report -par-report)
add_definitions( -DUSE_DOUBLE_VEC3 -DUSE_KDTREE ) 


#INCLUDE_DIRECTORIES($ENV{EXTERNAL_INCLUDE_DIRS})

#GET_DIRECTORY_PROPERTY(incdirs INCLUDE_DIRECTORIES) 
#MESSAGE ("INCLUDE_DIRECTORIES: ${incdirs}")

# Make sure we build library and tests subtrees
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/src)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/tests)

