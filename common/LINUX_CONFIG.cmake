MESSAGE(STATUS "LINUX detected.")

IF (CUDA_FOUND) 
	#	SET(CUDA_NVCC_FLAGS "-arch=compute_13;-code=sm_13")
	SET(CUDA_NVCC_FLAGS "-ccbin;icc;-arch=sm_13")
    #	SET(CUDA_NVCC_FLAGS "-arch=sm_13")
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-D__builtin_stdarg_start=__builtin_va_start")
	MESSAGE(STATUS "CUDA NVCCFLAGS SET: ${CUDA_NVCC_FLAGS}")
ENDIF (CUDA_FOUND)

# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (
)


# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES 
)

