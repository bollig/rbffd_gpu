
###############################################
# External includes we need on the include path
###############################################
	
	# NOTE: we need the ; here or cmake will not interpret the line properly
	SET (ENV{EXTERNAL_INCLUDE_DIRS} 
		"$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}"
	#	"$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/cusp;${CMAKE_CURRENT_SOURCE_DIR}/thrust"
	)

	MESSAGE(STATUS "EXTERNAL CUDA LIBRARIES ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")
		
