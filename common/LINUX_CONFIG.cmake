MESSAGE(STATUS "LINUX detected.")

get_filename_component(_compiler ${CMAKE_CXX_COMPILER} NAME)

IF (${_compiler} STREQUAL "icpc")
    add_definitions(-opt-report -par-report -vec-report ) 
    SET(CUDA_NVCC_FLAGS "-ccbin;icc;-arch=sm_13")
ELSE ()
    # ASSUME WE'RE USING GCC
    add_definitions(-ftree-vectorizer-verbose=2 ) 
    set(USE_ICC OFF)

    SITE_NAME(_hostname)
    if (${_hostname} MATCHES "kid.*")
        FIND_LIBRARY (iomp5 iomp5 PATHS
            /opt/intel/composerxe-2011.5.220/compiler/lib/intel64/
            )
        MESSAGE(STATUS "Found iomp5 in: ${iomp5}")
        set (ADDITIONAL_REQUIRED_LIBRARIES "${ADDITIONAL_REQUIRED_LIBRARIES};${iomp5}")
    ENDIF(${_hostname} MATCHES "kid.*")
    	
    SET(CUDA_NVCC_FLAGS "-ccbin;g++;-arch=sm_13")

ENDIF ()


IF (CUDA_FOUND) 
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-D__builtin_stdarg_start=__builtin_va_start")
	MESSAGE(STATUS "CUDA NVCCFLAGS SET: ${CUDA_NVCC_FLAGS}")
ENDIF (CUDA_FOUND)


# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (
    # RHEL location for openmpi
    /usr/include/openmpi-x86_64/
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (
    ${ICC_LINK_DIR}
)


# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES
    ${ADDITIONAL_REQUIRED_LIBRARIES}
)

