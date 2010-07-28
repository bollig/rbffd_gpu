
###############################################
# External includes we need on the include path
###############################################
	
	# NOTE: we need the ; here or cmake will not interpret the line properly
	SET (ENV{EXTERNAL_INCLUDE_DIRS} 
		"$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/ViennaCL-1.0.4"
	)

	MESSAGE(STATUS "EXTERNAL OpenCL LIBRARIES ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")
		
