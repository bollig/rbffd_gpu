FIND_PACKAGE(CUDA 4.0 REQUIRED)	

if (NOT CUDA_FOUND) 
	MESSAGE(WARN "CUDA NOT FOUND")
else () 
	MESSAGE(STATUS "CUDA VERSION ${CUDA_VERSION} FOUND")
	if (${CUDA_VERSION} VERSION_GREATER "4.0")
#STRCMP "${CUDA_VERSION}" "4.0")
	set (cusp_directory "cusp-v0.3.1")
else() 
	set (cusp_directory "cusp-v0.2")
	endif()
endif ()

	MESSAGE(STATUS "Using ${cusp_directory}")

###############################################
# External includes we need on the include path
###############################################

# NOTE: we need the ; here or cmake will not interpret the line properly
	SET (ENV{EXTERNAL_INCLUDE_DIRS} 
			# Add cusp
			"$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/${cusp_directory}"
			# IF necessary add thrust (now we have thrust
			# included in CUDA driver we dont need this. 
#	"$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/cusp;${CMAKE_CURRENT_SOURCE_DIR}/thrust"
	    )

	MESSAGE(STATUS "EXTERNAL CUDA LIBRARIES ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")

