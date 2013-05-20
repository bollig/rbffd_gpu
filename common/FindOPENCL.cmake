# FROM: http://forums.nvidia.com/index.php?showtopic=97795
# Written by theMarix
# Modified by Ian Johnson, Gordon Erlebacher and Evan Bollig Summer 2010

# - Try to find OpenCL
# Once done this will define
#  
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIR  - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL
#
# WIN32 should work, but is untested

# Drop the vars from cache or the NVIDIA libs on kirk
# will not be available on troi and vice versa.
unset(OPENCL_INCLUDE_DIR CACHE)
unset(OPENCL_LIBRARIES CACHE)

#NOTE: NOT SURE IF THIS WILL WORK IN VISUAL STUDIO
IF (WIN32 OR CYGWIN)

	# NOTE: if CUDA_PATH is not set on the user environment we can assume we're
	# looking for ATI OpenCL
	SET (CUDA_PATH $ENV{CUDA_PATH})

	IF(CUDA_PATH)
		MESSAGE("IN NVIDIA CASE!")

		FILE(TO_CMAKE_PATH $ENV{CUDA_PATH} CONVERTED_PATH_LIST)

		# Remove the leading drive (i.e., "C") and make it lower case
		# for cygwin
		LIST(GET CONVERTED_PATH_LIST 0 WIN_DRIVE)
		LIST(GET CONVERTED_PATH_LIST 1 WIN_PATH)
		STRING(TOLOWER ${WIN_DRIVE} WIN_DRIVE_LOWER)

		SET(GPU_TOOLKIT "/cygwin/${WIN_DRIVE_LOWER}${WIN_PATH}")

	ELSE (CUDA_PATH)
	# TODO: ATI  
		MESSAGE(ERROR "IN ATI CASE! Not implemented.")
		SET (GPU_TOOLKIT "/cygdrive/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v3.2/")
	ENDIF (CUDA_PATH) 

	FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h 
			$ENV{GPU_COMPUTING_TOOLKIT}/include
		 )


	MESSAGE("GPU_TOOLKIT/include= ${GPU_TOOLKIT}/include") 

	FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h 
			PATHS 
			${GPU_TOOLKIT}/include    
			$ENV{CUDA_PATH}/include
		 )

	# TODO this is only a hack assuming the 64 bit library will
	# not be found on 32 bit system
	FIND_LIBRARY(OPENCL_LIBRARIES 
			NAMES OpenCL opencl64 
			PATHS 
			$ENV{OPENCL_ROOT}/lib
			$ENV{OPENCL_ROOT}/lib64
			/cygdrive/c/Windows/SysWOW64
			/cygdrive/c/Windows/System32
			${GPU_TOOLKIT}/lib/Win32
			${GPU_TOOLKIT}/lib/x64
			$ENV{CUDA_PATH}/lib/Win32
			$ENV{CUDA_PATH}/lib64
			$ENV{CUDA_PATH}/lib
			~/local/lib
			NO_DEFAULT_PATH
		    )
	IF( OPENCL_LIBRARIES )
		FIND_LIBRARY(OPENCL_LIBRARIES 
				NAMES OpenCL opencl32
				PATHS 
				$ENV{OPENCL_ROOT}/lib
				$ENV{OPENCL_ROOT}/lib32
				/cygdrive/c/Windows/SysWOW64
				/cygdrive/c/Windows/System32
				${GPU_TOOLKIT}/lib/Win32
				${GPU_TOOLKIT}/lib/x64
				$ENV{CUDA_PATH}/lib/Win32
				$ENV{CUDA_PATH}/lib/x64
				$ENV{CUDA_PATH}/lib64
				$ENV{CUDA_PATH}/lib
				~/local/lib
				NO_DEFAULT_PATH
			    )
	ENDIF( OPENCL_LIBRARIES )

ELSE (WIN32 OR CYGWIN)

# Unix style platforms
# We also search for OpenCL in the NVIDIA GPU SDK default location
#SET(OPENCL_INCLUDE_DIR  "$ENV{OPENCL_HOME}/common/inc"
#   CACHE PATH "path to Opencl Include files")

#message(***** OPENCL_INCLUDE_DIR: "${OPENCL_INCLUDE_DIR}" ********)

# does not work. WHY? 
#SET(inc  $ENV{CUDA_LOCAL}/../OpenCL/common/inc /usr/include)
	FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h PATHS
			$ENV{EXTERNAL_INCLUDE_DIRS}
			"/Developer/GPU Computing/OpenCL/common/inc/" 
			"/Developer/GPU_Computing/OpenCL/common/inc/" 
			~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ 
			$ENV{OPENCL_ROOT}/include
			$ENV{ATISTREAMSDKROOT}/include
			$ENV{CUDA_INSTALL_PATH}/include
			/usr/include 
		 )

	FIND_LIBRARY(OPENCL_LIBRARIES OpenCL ENV 
			$ENV{OPENCL_ROOT}/lib64
			$ENV{OPENCL_ROOT}/lib
			LD_LIBRARY_PATH
			NO_DEFAULT_PATH
		    )

	IF (USE_ICC)
		FIND_LIBRARY(INTEL_OCL_LIBRARIES intelocl ENV
				$ENV{OPENCL_ROOT}/lib64
				$ENV{OPENCL_ROOT}/lib
				LD_LIBRARY_PATH
				NO_DEFAULT_PATH
			)
		if (INTEL_OCL_LIBRARIES)
			message(STATUS "INTELOCL Library: ${INTEL_OCL_LIBRARIES}")
			set (OPENCL_LIBRARIES ${INTEL_OCL_LIBRARIES} ${OPENCL_LIBRARIES})
		endif()
	ENDIF()

	if (OPENCL_LIBRARIES)
		message(STATUS "OPENCL library is on LD_LIBRARY_PATH")
	else()
		FIND_LIBRARY(OPENCL_LIBRARIES OpenCL ENV 
				# Include the default search path here
				LD_LIBRARY_PATH
			    )
	endif()
	message(STATUS "OPENCL_INCLUDE_DIR:  ${OPENCL_INCLUDE_DIR}")
	message(STATUS "OPENCL_LIBRARIES:  ${OPENCL_LIBRARIES}")
#message(***** OPENCL ENV: "$ENV{GPU_SDK}" ********)

#~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ 


ENDIF (WIN32 OR CYGWIN)

	SET( OPENCL_FOUND "NO" )
IF(OPENCL_LIBRARIES )
	SET( OPENCL_FOUND "YES" )
ENDIF(OPENCL_LIBRARIES)

	MARK_AS_ADVANCED(
			OPENCL_INCLUDE_DIR
			)
