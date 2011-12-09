PROJECT(ViennaCLTests)

cmake_minimum_required(VERSION 2.4)

#-----------------------------------------------------------------------------
#specify include and source directory
INCLUDE_DIRECTORIES("../")    # ViennaCL root

# alternate boost path for macs
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   INCLUDE_DIRECTORIES("/opt/local/include") 
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# **************************  Section 2: Configure OpenCL **************************

#
# AMD APP SDK:
# add include and lib path to build environment
# The linker will most likely pick the correct link directory
# Otherwise, feel free to comment the wrong link directory for your machine.
#
# The following lines can be commented if AMD APP SDK is not in use.
#
SET(AppSDK $ENV{AMDAPPSDKROOT})
IF(AppSDK)
 INCLUDE_DIRECTORIES($ENV{AMDAPPSDKROOT}/include) 
 LINK_DIRECTORIES($ENV{AMDAPPSDKROOT}/lib/x86)       #use this on 32 bit systems
 LINK_DIRECTORIES($ENV{AMDAPPSDKROOT}/lib/x86_64)    #use this on 64 bit systems
ENDIF(AppSDK)
#
# Note that the environment variable was ATISTREAMSDKROOT in older versions of the SDK:
#
SET(StreamSDK $ENV{ATISTREAMSDKROOT})
IF(StreamSDK)
 INCLUDE_DIRECTORIES($ENV{ATISTREAMSDKROOT}/include) 
 LINK_DIRECTORIES($ENV{ATISTREAMSDKROOT}/lib/x86)       #use this on 32 bit systems
 LINK_DIRECTORIES($ENV{ATISTREAMSDKROOT}/lib/x86_64)    #use this on 64 bit systems
ENDIF(StreamSDK)


#
# Intel OpenCL SDK:
# add include and lib path to build environment
# The linker will most likely pick the correct link directory
# Otherwise, feel free to comment the wrong link directory for your machine.
#
# The following lines can be commented if Intel OpenCL SDK is not in use.
#
SET(IntelSDK $ENV{INTELOCLSDKROOT})
IF(IntelSDK)
 INCLUDE_DIRECTORIES($ENV{INTELOCLSDKROOT}/include) 
 LINK_DIRECTORIES($ENV{INTELOCLSDKROOT}/lib/x86)       #use this on 32 bit systems
 LINK_DIRECTORIES($ENV{INTELOCLSDKROOT}/lib/x64)    #use this on 64 bit systems
ENDIF(IntelSDK)

#
# For NVIDIA hardware, there should be no need for additional configurations at this point
#

#-----------------------------------------------------------------------------

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
SET(PROGS scalar vector matrix sparse iterators blas3)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
FOREACH(PROG ${PROGS})
   ADD_EXECUTABLE(${PROG} ../src/${PROG}.cpp)
   ADD_test(${PROG} ${PROG})
   TARGET_LINK_LIBRARIES(${PROG} OpenCL)
ENDFOREACH(PROG)

ADD_EXECUTABLE(external_linkage  ../src/external_1.cpp
                                 ../src/external_2.cpp)

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
