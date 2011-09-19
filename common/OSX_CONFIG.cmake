FIND_LIBRARY (accelerate NAMES Accelerate)
# CLAPACK is available from netlib.org or linux repositories
# Use Accelerate library on OSX
FIND_LIBRARY (clapack clapack PATHS /usr/lib) 

MESSAGE(STATUS "MacOS X detected. Added '-framework Accelerate' to compiler flags")


IF (CUDA_FOUND) 
	# THESE ARE FOR OSX and older versions of CUDA (i think <= 3.0) 
	#SET(CUDA_NVCC_FLAGS "-Xcompiler;-D__builtin_stdarg_start=__builtin_va_start")
	#MESSAGE("CUDA NVCCFLAGS SET: ${CUDA_NVCC_FLAGS}")
ENDIF (CUDA_FOUND)


# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (
    #/sw/include
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (
    #/sw/lib
)
	
# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES 
	${clapack}
	${accelerate}
)

