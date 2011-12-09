PROJECT(ViennaCL)

cmake_minimum_required(VERSION 2.6)

#if you don't want the full compiler output, remove or comment the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)


# ************************** Section 1: Configure external dependencies **************************

#
# If you are interested in the impact of different kernel parameters on performance,
# you may want to give ViennaProfiler a try (see http://sourceforge.net/projects/viennaprofiler/)
# Set your connection parameters in examples/parameters/common_vprof.hpp accordingly.
#
#SET(ENABLE_VIENNAPROFILER ON)
#SET(MYSQLPATH "/opt/boost")        #path to MySQL header files
#SET(MYSQLPPPATH "/opt/boost")        #path to MySQL++ header files


#
# If you want to build the examples that use boost::numeric::ublas, enable the following:
#
#SET(ENABLE_UBLAS ON)
#SET(BOOSTPATH "/opt/boost")             #put your path to boost here if not in global include folder

#
# alternate boost path for macs
#
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   INCLUDE_DIRECTORIES("/opt/local/include") 
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")



#
# If you want to build the examples that use Eigen, uncomment the following:
#
#SET(ENABLE_EIGEN ON)
#SET(EIGENPATH "/path/to/eigen/")  #put your path to Eigen here if not in global include folder


#
# If you want to build the examples that use MTL4, uncomment the following:
#
#SET(ENABLE_MTL4 ON)
#SET(MTL4PATH "/path/to/MTL-4/")  #put your path to MTL4 here if not in global include folder



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


# ************************** Section 3: ViennaCL standalone builds **************************

#
# Specify include and source directory
#
INCLUDE_DIRECTORIES(".")
INCLUDE_DIRECTORIES("external/")


#add definitions, compiler switches, etc.
IF(DEFINED CMAKE_BUILD_TYPE)
 SET (CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
ELSE()
 SET (CMAKE_BUILD_TYPE Release)
ENDIF()

#
# Set high warning level on GCC and enable optimizations (or debug)
#
IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -O0 -g)             #debug build
   ADD_DEFINITIONS(-Wall -pedantic -O3)       #release build
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

#
# Standalone ViennaCL executables
#

#tutorials:
ADD_EXECUTABLE(blas1 examples/tutorial/blas1.cpp)
ADD_EXECUTABLE(custom-kernels examples/tutorial/custom-kernels.cpp)
ADD_EXECUTABLE(custom-context examples/tutorial/custom-context.cpp)
ADD_EXECUTABLE(viennacl-info examples/tutorial/viennacl-info.cpp)

#benchmarks
ADD_EXECUTABLE(vectorbench examples/benchmarks/vector.cpp)
ADD_EXECUTABLE(openclbench examples/benchmarks/opencl.cpp)
ADD_EXECUTABLE(blas3bench examples/benchmarks/blas3.cpp)

#parameters:
ADD_EXECUTABLE(vectorparams examples/parameters/vector.cpp
                            external/pugixml/src/pugixml.cpp)
ADD_EXECUTABLE(matrixparams examples/parameters/matrix.cpp
                            external/pugixml/src/pugixml.cpp)
ADD_EXECUTABLE(sparseparams examples/parameters/sparse.cpp
                            external/pugixml/src/pugixml.cpp)
ADD_EXECUTABLE(parameter_reader examples/parameters/parameter_reader.cpp
                                external/pugixml/src/pugixml.cpp)                        

#
# Mac OS X specific linker part
#
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#
# Linux and Windows specific linker part
#
IF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
   #tutorials
   TARGET_LINK_LIBRARIES(blas1 OpenCL)
   TARGET_LINK_LIBRARIES(custom-kernels OpenCL)
   TARGET_LINK_LIBRARIES(custom-context OpenCL)
   TARGET_LINK_LIBRARIES(viennacl-info OpenCL)

   #benchmarks
   TARGET_LINK_LIBRARIES(vectorbench OpenCL)
   TARGET_LINK_LIBRARIES(openclbench OpenCL)
   TARGET_LINK_LIBRARIES(blas3bench OpenCL)

   #parameter estimation
   TARGET_LINK_LIBRARIES(vectorparams OpenCL)
   TARGET_LINK_LIBRARIES(matrixparams OpenCL)
   TARGET_LINK_LIBRARIES(sparseparams OpenCL)
   TARGET_LINK_LIBRARIES(parameter_reader OpenCL)   
ENDIF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))


# ************************** Section 4: ViennaCL with external libraries **************************

#
# ViennaProfiler related configuration
#
IF(DEFINED ENABLE_VIENNAPROFILER)
   INCLUDE_DIRECTORIES(${MYSQLPATH})
   INCLUDE_DIRECTORIES(${MYSQLPPPATH})

   #vector:
   ADD_EXECUTABLE(vectorparams_vprof examples/parameters/vector.cpp
                                     external/pugixml/src/pugixml.cpp)
   SET_PROPERTY(TARGET vectorparams_vprof PROPERTY COMPILE_DEFINITIONS ENABLE_VIENNAPROFILER)
   TARGET_LINK_LIBRARIES(vectorparams_vprof OpenCL mysqlpp)

   #dense matrix:
   ADD_EXECUTABLE(matrixparams_vprof examples/parameters/matrix.cpp
                                     external/pugixml/src/pugixml.cpp)
   SET_PROPERTY(TARGET matrixparams_vprof PROPERTY COMPILE_DEFINITIONS ENABLE_VIENNAPROFILER)
   TARGET_LINK_LIBRARIES(matrixparams_vprof OpenCL mysqlpp)

   #sparse matrix:
   ADD_EXECUTABLE(sparseparams_vprof examples/parameters/sparse.cpp
                                     external/pugixml/src/pugixml.cpp)
   SET_PROPERTY(TARGET sparseparams_vprof PROPERTY COMPILE_DEFINITIONS ENABLE_VIENNAPROFILER)
   TARGET_LINK_LIBRARIES(sparseparams_vprof OpenCL mysqlpp)
ENDIF(DEFINED ENABLE_VIENNAPROFILER)


#
# ublas related configuration
#
IF(DEFINED ENABLE_UBLAS)
  INCLUDE_DIRECTORIES(${BOOSTPATH})
  
  #tutorials:
  ADD_EXECUTABLE(blas2 examples/tutorial/blas2.cpp) 
  ADD_EXECUTABLE(blas3 examples/tutorial/blas3.cpp)
  ADD_EXECUTABLE(iterative examples/tutorial/iterative.cpp)
  ADD_EXECUTABLE(iterative-ublas examples/tutorial/iterative-ublas.cpp)

  #benchmarks:
  ADD_EXECUTABLE(sparsebench examples/benchmarks/sparse.cpp)
  ADD_EXECUTABLE(solverbench examples/benchmarks/solver.cpp)

  IF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    TARGET_LINK_LIBRARIES(iterative OpenCL)
    TARGET_LINK_LIBRARIES(blas2 OpenCL)
    TARGET_LINK_LIBRARIES(blas3 OpenCL)

    TARGET_LINK_LIBRARIES(sparsebench OpenCL)
    TARGET_LINK_LIBRARIES(solverbench OpenCL)
  ENDIF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))

ENDIF(DEFINED ENABLE_UBLAS)



#
# Eigen related configuration
#
IF(DEFINED ENABLE_EIGEN)
  INCLUDE_DIRECTORIES(${EIGENPATH})
  ADD_EXECUTABLE(iterative-eigen examples/tutorial/iterative-eigen.cpp)
  ADD_EXECUTABLE(eigen-with-viennacl examples/tutorial/eigen-with-viennacl.cpp)
  
  IF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    TARGET_LINK_LIBRARIES(eigen-with-viennacl OpenCL)
  ENDIF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
ENDIF(DEFINED ENABLE_EIGEN)



#
# MTL4 related configuration
#
IF(DEFINED ENABLE_MTL4)
  INCLUDE_DIRECTORIES(${MTL4PATH})
  ADD_EXECUTABLE(iterative-mtl4 examples/tutorial/iterative-mtl4.cpp)
  ADD_EXECUTABLE(mtl4-with-viennacl examples/tutorial/mtl4-with-viennacl.cpp)

  IF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
    TARGET_LINK_LIBRARIES(mtl4-with-viennacl OpenCL)
  ENDIF((${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows"))
ENDIF(DEFINED ENABLE_MTL4)
