###############################################
# Build Options (Definitions and compiler flags)
###############################################
# Used by ALL compilers
#ADD_DEFINITIONS(-g -O3)
ADD_DEFINITIONS(-O3)


###############################################
# EXTENSIONS TO INCLUDE: 
###############################################
ENABLE_TESTING()
INCLUDE (CPack)
INCLUDE (UseDoxygen)
FIND_PACKAGE (MPI)

# Let Armadillo find MKL stuff
OPTION (USE_MKL "Enable/Disable the linking to MKL drop in replacements to BLAS and LAPACK" ON)
if (CMAKE_C_COMPILER MATCHES "icc") 
	
if (USE_MKL) 
    include(ARMA_FindMKL)
    message(STATUS "MKL_FOUND     = ${MKL_FOUND}")
    include_directories(${MKL_INCLUDE_DIRS})
    SET (ENV{EXTERNAL_LIB_DEPENDENCIES} "$ENV{EXTERNAL_LIB_DEPENDENCIES};${MKL_LIBRARIES}")
endif(USE_MKL)
else () 
	message ("Disabled MKL support. You are not using an intel compiler.")
endif ()


IF (NOT USE_OPENCL)
    set (OPENCL_FOUND false)
ELSE ()
    FIND_PACKAGE (OPENCL)
ENDIF ()


#IF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})
#    MESSAGE (WARNING "\nWARNING! Dep: libcuda was NOT FOUND. Disabling cuda support for framework. Please install NVidia proprietary driver and verify you have an NVidia GPU if you want it enabled.\n")
#    set(CUDA_FOUND false)
#ENDIF (CUDA_FOUND AND NOT EXISTS ${CUDA_CUDA_LIBRARY})
IF (USE_CUDA)
    unset(CUDA_CUDA_LIBRARY CACHE)
    FIND_PACKAGE (CUDA)
ELSE () 
    set (CUDA_FOUND false)
ENDIF ()



###############################################
# Setup MPI 
###############################################

IF (MPI_FOUND AND USE_MPI)
    SET (CMAKE_CXX_FLAGS ${MPI_COMPILE_FLAGS})
    SET (CMAKE_C_FLAGS ${MPI_COMPILE_FLAGS})
    SET (CMAKE_LINK_FLAGS ${MPI_LINK_FLAGS})

    INCLUDE_DIRECTORIES (${MPI_INCLUDE_PATH})
    # NOTE: add a target_link_library( MPI_LIBRARIES) for libs and bins
    # TESTS that run parallel should use MPIEXEC
ENDIF (MPI_FOUND AND USE_MPI) 



# This makes finding boost more robust when we have custom installs
set ( Boost_NO_BOOST_CMAKE  true ) 
set ( Boost_NO_SYSTEM_PATHS true )
set ( BOOST_MIN_VERSION     1.48.0)

# This guarantees geometry and other features we use will exist.
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS filesystem system REQUIRED)

if(Boost_FOUND)
  
  message(STATUS "Boost_MAJOR_VERSION = ${Boost_MAJOR_VERSION}")
  message(STATUS "Boost_MINOR_VERSION = ${Boost_MINOR_VERSION}")
  
  message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
  include_directories(BEFORE ${Boost_INCLUDE_DIRS})
  #SET (ENV{EXTERNAL_LIB_DEPENDENCIES} "$ENV{EXTERNAL_LIB_DEPENDENCIES};${Boost_LIBRARIES}")
#   if(Boost_FOUND)
#      include_directories(${Boost_INCLUDE_DIRS})
#      add_executable(foo foo.cc)
#      target_link_libraries(foo ${Boost_LIBRARIES})
#   endif()
endif()



###############################################
# Locate Required Libraries
###############################################
# Find library: find_library(<VAR> name1 [path1 path2 ...])
if (DEFINED ENV{ARMADILLO_ROOT})
    set(ARMADILLO_INCLUDE "${ARMADILLO_INCLUDE};$ENV{ARMADILLO_ROOT}/include" )
    set(ARMADILLO_LIB_PATH "${ARMADILLO_LIB_PATH};$ENV{ARMADILLO_ROOT}/lib;$ENV{ARMADILLO_ROOT}/lib64")
endif()

if (DEFINED ENV{ARMADILLOROOT})
    set(ARMADILLO_INCLUDE "${ARMADILLO_INCLUDE};$ENV{ARMADILLOROOT}/include" )
    set(ARMADILLO_LIB_PATH "${ARMADILLO_LIB_PATH};$ENV{ARMADILLOROOT}/lib;$ENV{ARMADILLOROOT}/lib64")
endif()


if (DEFINED ARMADILLO_ROOT )
    set(ARMADILLO_INCLUDE "${ARMADILLO_ROOT}/include;${ARMADILLO_INCLUDE}" )
    set(ARMADILLO_LIB_PATH "${ARMADILLO_ROOT}/lib;${ARMADILLO_ROOT}/lib64;${ARMADILLO_LIB_PATH}")
endif()
if (DEFINED ARMADILLOROOT )
    set(ARMADILLO_INCLUDE "${ARMADILLOROOT}/include;${ARMADILLO_INCLUDE}" )
    set(ARMADILLO_LIB_PATH "${ARMADILLOROOT}/lib;${ARMADILLOROOT}/lib64;${ARMADILLO_LIB_PATH}")
endif() 

MESSAGE(STATUS "SEARCHING FOR ARMADILLO IN LIB PATH: ${ARMADILLO_LIB_PATH}")


# Download and install Armadillo separately. 
# Specify local installation dir here. If installed globally the dir is unnecessary.
FIND_LIBRARY (armadillo armadillo PATHS
    # efb06: bollig account
    ${ARMADILLO_LIB_PATH}
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
if (DEFINED armadillo) 
    SET(ARMADILLO_FOUND true)
endif()

if (ARMADILLO_FOUND)
 MESSAGE(STATUS "Found armadillo in: ${armadillo} (Include: ${ARMADILLO_INCLUDE})")
 include_directories( ${ARMADILLO_INCLUDE} )
else() 
    MESSAGE(FATAL_ERROR " Armadillo Not Found! ")
endif() 

#SET(CMAKE_FIND_LIBRARY_SUFFIXES_SAVED ${CMAKE_FIND_LIBRARY_SUFFIXES}) #Backup
#LIST(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.3")


if (DEFINED ENV{FFTW_ROOT})
    set(FFTW_INCLUDE "${FFTW_INCLUDE};$ENV{FFTW_ROOT}/include" )
    set(FFTW_LIB_PATH "${FFTW_LIB_PATH};$ENV{FFTW_ROOT}/lib;$ENV{FFTW_ROOT}/lib64")
endif()

if (DEFINED ENV{FFTWROOT})
    set(FFTW_INCLUDE "${FFTW_INCLUDE};$ENV{FFTWROOT}/include" )
    set(FFTW_LIB_PATH "${FFTW_LIB_PATH};$ENV{FFTWROOT}/lib;$ENV{FFTWROOT}/lib64")
endif()


if (DEFINED FFTW_ROOT )
    set(FFTW_INCLUDE "${FFTW_ROOT}/include;${FFTW_INCLUDE}" )
    set(FFTW_LIB_PATH "${FFTW_ROOT}/lib;${FFTW_ROOT}/lib64;${FFTW_LIB_PATH}")
endif()
if (DEFINED FFTWROOT )
    set(FFTW_INCLUDE "${FFTWROOT}/include;${FFTW_INCLUDE}" )
    set(FFTW_LIB_PATH "${FFTWROOT}/lib;${FFTWROOT}/lib64;${FFTW_LIB_PATH}")
endif() 

MESSAGE(STATUS "SEARCHING FOR FFTW3 IN LIB PATH: ${FFTW_LIB_PATH}")


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
    ${FFTW_LIB_PATH}
    #	NO_DEFAULT_PATH
    )
SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVED}) #Restore
MESSAGE(STATUS "Found fftw3 in: ${fftw3} (Include: ${FFTW_INCLUDE}")
include_directories(AFTER ${FFTW_INCLUDE} )


###############################################
# Setup GPU APIs
###############################################


###############################################
# External dependency search paths
###############################################
# Directories searched for headers ORDER does not matter. 
# If a directory does not exist it is skipped
#SET (RBF_INCLUDE_DIRS 
#    .
#    /Users/erlebach/Documents/src/fftw-3.2.2/include
#    /Users/erlebach/Documents/src/armadillo-0.9.52/include
#    $ENV{INCLUDE_PATH}
#    /opt/local/include   # for boost
#    ~/local/include
#    ~/local/usr/include
#    ${ARMADILLO_INCLUDE}
#    )

#INCLUDE_DIRECTORIES ( ${RBF_INCLUDE_DIRS} )
