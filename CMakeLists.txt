PROJECT(RBF_framework)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# SOME HARD-CODED PATHS NECESSARY: 
SET(GLIB ~/gordon_tests)

SET (FRAMEWORK_LIBRARY RBF)
SET (FRAMEWORK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

################ BEGIN OS DEPENDENT CONFIGS ###################
	
SET( CMAKE_MODULE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/common
	${CMAKE_CURRENT_SOURCE_DIR}/common/UseDoxygen
	${CMAKE_CURRENT_SOURCE_DIR}/common/trilinos_cmake
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


# Make sure we build library and tests subtrees
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/src)
ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/tests)
