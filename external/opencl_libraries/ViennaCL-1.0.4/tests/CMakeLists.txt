PROJECT(ViennaCLTests)

cmake_minimum_required(VERSION 2.4)

#-----------------------------------------------------------------------------
#specify include and source directory
INCLUDE_DIRECTORIES("../")    # ViennaCL root

# alternate boost path for macs
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   INCLUDE_DIRECTORIES("/opt/local/include") 
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# ATI stream sdk:
# add include and lib path to build environment
SET(OPENCLPATH $ENV{ATISTREAMSDKROOT})
INCLUDE_DIRECTORIES(${OPENCLPATH}/include) 
LINK_DIRECTORIES(${OPENCLPATH}/lib/x86_64) 

#add definitions, compiler switches, etc.
IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -O0 -g)
   ADD_DEFINITIONS(-Wall -pedantic -O3)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Configure Dart testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
INCLUDE(CTest)
ENABLE_TESTING()
IF(BUILD_TESTING)
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
ENDIF(BUILD_TESTING)

#-----------------------------------------------------------------------------
# add the source files which should be tested without the trailing *.cpp
SET(PROGS scalar vector matrix sparse)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
FOREACH(PROG ${PROGS})
   ADD_EXECUTABLE(${PROG} ../src/${PROG}.cpp)
   ADD_test(${PROG} ${PROG})
ENDFOREACH(PROG)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Mac OS X specific linker part
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Linux specific linker part
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
   set(CMAKE_EXE_LINKER_FLAGS "-lOpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
#-----------------------------------------------------------------------------
