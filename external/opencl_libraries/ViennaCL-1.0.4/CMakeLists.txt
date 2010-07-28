PROJECT(ViennaCL)

cmake_minimum_required(VERSION 2.4)

#if you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)

#specify include and source directory
INCLUDE_DIRECTORIES(".")

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
IF(DEFINED CMAKE_BUILD_TYPE)
 SET (CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
ELSE()
 SET (CMAKE_BUILD_TYPE Release)
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -O0 -g)
   ADD_DEFINITIONS(-Wall -pedantic -O3)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

#list all source files here
ADD_EXECUTABLE(tut1 examples/tutorial/tut1.cpp)
ADD_EXECUTABLE(tut2 examples/tutorial/tut2.cpp)
ADD_EXECUTABLE(tut3 examples/tutorial/tut3.cpp)
ADD_EXECUTABLE(tut4 examples/tutorial/tut4.cpp)
ADD_EXECUTABLE(tut5 examples/tutorial/tut5.cpp)

ADD_EXECUTABLE(vectorbench examples/benchmarks/vector.cpp)
ADD_EXECUTABLE(sparsebench examples/benchmarks/sparse.cpp)
ADD_EXECUTABLE(solverbench examples/benchmarks/solver.cpp)

# Mac OS X specific linker part
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Linux specific linker part
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
   TARGET_LINK_LIBRARIES(tut1 OpenCL)
   TARGET_LINK_LIBRARIES(tut2 OpenCL)
   TARGET_LINK_LIBRARIES(tut3 OpenCL)
   #note: no OpenCL for tut4
   TARGET_LINK_LIBRARIES(tut5 OpenCL)

   TARGET_LINK_LIBRARIES(vectorbench OpenCL)
   TARGET_LINK_LIBRARIES(sparsebench OpenCL)
   TARGET_LINK_LIBRARIES(solverbench OpenCL)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# Windows specific linker part
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   TARGET_LINK_LIBRARIES(tut1 OpenCL)
   TARGET_LINK_LIBRARIES(tut2 OpenCL)
   TARGET_LINK_LIBRARIES(tut3 OpenCL)
   #note: no OpenCL for tut4
   TARGET_LINK_LIBRARIES(tut5 OpenCL)

   TARGET_LINK_LIBRARIES(vectorbench OpenCL)
   TARGET_LINK_LIBRARIES(sparsebench OpenCL)
   TARGET_LINK_LIBRARIES(solverbench OpenCL)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

