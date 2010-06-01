PROJECT(RBF_framework)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

################ BEGIN OS DEPENDENT CONFIGS ###################
	
SET( CMAKE_MODULE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/common
	${CMAKE_CURRENT_SOURCE_DIR}/common/UseDoxygen
)


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



# Get common build instructions used by all operating systems
INCLUDE (COMMON_CONFIG)

# Setup any external libraries we might need
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/external)

# These must come AFTER the external subdir so we have 
# access to CMake variables for each dependency
SET (FRAMEWORK_LIBRARY RBF)	# Name of the framework lib (i.e., lib<name>.dylib)
SET (FRAMEWORK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# This is an external dependency
# Its name should not change
SET( FRAMEWORK_DEPENDENCIES
	utilities
        timing
)

# This is a neat trick: since the external deps are now part of
# this source tree we can reference their subtree as ${<name>_SOURCE_DIR}
# so long as CMake is told that they are subdirs (i.e., add_subdirectory) 
SET( FRAMEWORK_DEP_INCLUDE_DIRS
	${utilities_SOURCE_DIR}
        ${timing_SOURCE_DIR}
)

# Make sure we build library and tests subtrees
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/src)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/tests)

