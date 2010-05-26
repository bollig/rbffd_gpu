# Environment variables
#	GRAPHIC_LIBS_HOME/  : subdirectories include graphic_utils, utilities, ...
#	CUDA_HOME : root of cuda installation: (e.g., /usr/local/cuda, under which
#	     one has bin/, ...)
#	CUDA_LOCAL : root of NVIDIA_SDK, under which one finds common/, lib/, #	bin/, ...


project(utilities)
cmake_minimum_required(VERSION 2.6)

set(utils cu.cpp array_cuda_1d.cpp array_cuda_t.cpp)

set(graphics_libs $ENV{GRAPHIC_LIBS_HOME})

#for macs
#ADD_DEFINITIONS(-arch i386)
#for linux
#ADD_DEFINITIONS(-arch x86_64)
ADD_DEFINITIONS(-m64)

Include_directories( 
    . 
	/Users/Documents/src/graphics_libs/utilities
	${graphics_libs}/utilities
	$ENV{CUDA_HOME}/include 
	$ENV{CUDA_LOCAL}/common/inc 
)

set (libraries ${GRAPHIC_LIBS_HOME}/utilities)

add_library(cuda_utilities STATIC ${utils})


#	../utilities
