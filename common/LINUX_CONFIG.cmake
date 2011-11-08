MESSAGE(STATUS "LINUX detected.")

IF (CUDA_FOUND) 
    get_filename_component(_compiler ${CMAKE_CXX_COMPILER} NAME)
    IF (${_compiler} STREQUAL "icpc")
        add_definitions(-opt-report -par-report -vec-report ) 
    	SET(CUDA_NVCC_FLAGS "-ccbin;icc;-arch=sm_13")
    ELSE ()
        # ASSUME WE'RE USING GCC
        SITE_NAME(_hostname)
        if (${_hostname} MATCHES "kid.*")
            add_definitions(-ftree-vectorizer-verbose=2 ) 
            # We are on keeneland and need this library
            SET (ADDITIONAL_REQUIRED_LIBRARIES 
                #    /opt/intel/composerxe-2011.5.220/compiler/lib/intel64/libiomp5.so
                iomp5
            )
            SET (ICC_LINK_DIR
                /opt/intel/composerxe-2011.5.220/compiler/lib/intel64/
            )
        ENDIF(${_hostname} MATCHES "kid.*")
    ENDIF ()
    IF (USE_ICC) 
    	SET(CUDA_NVCC_FLAGS "-ccbin;icc;-arch=sm_13")
    ELSE (USE_ICC)
    	SET(CUDA_NVCC_FLAGS "-ccbin;g++;-arch=sm_13")
    ENDIF (USE_ICC)
    #	SET(CUDA_NVCC_FLAGS "-arch=sm_13")
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

