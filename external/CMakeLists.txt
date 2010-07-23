
###############################################
# External libraries we want to build
###############################################
	SET (EXTERNAL_LIBS
		graphics_lib
		kdtree
		cuda_libraries
	)

	# These libraries are provided by subdirectories and are 
	# required in the project. 
	SET (REQUIRED_EXTERNAL_LIBS 
		utilities	#graphics_lib/utilities
		timing		#grahpics_lib/timing
		kdtree		#kdtree/.
	)

	FOREACH (lib ${EXTERNAL_LIBS})
		ADD_SUBDIRECTORY (${lib})
	ENDFOREACH (lib)

	FOREACH (lib ${REQUIRED_EXTERNAL_LIBS})
		# add each subdir to a global cached variable for external include directories
		SET (ENV{EXTERNAL_INCLUDE_DIRS} "$ENV{EXTERNAL_INCLUDE_DIRS};${${lib}_SOURCE_DIR}")
		MESSAGE(STATUS "ADDED ${lib} SOURCE TO VAR: ENV{EXTERNAL_INCLUDE_DIRS} ")
	ENDFOREACH(lib)
