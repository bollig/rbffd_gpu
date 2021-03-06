
###############################################
# External libraries we want to build
###############################################
	SET (EXTERNAL_LIBS
		graphics_lib
		timers
		kdtree
        nist
	)

	# These libraries are provided by subdirectories and are 
	# required in the project. 
	SET (REQUIRED_EXTERNAL_LIBS 
		utilities	#graphics_lib/utilities
		timers		#grahpics_lib/timing
		kdtree		#kdtree/.
        nist
	)

	FOREACH (lib ${EXTERNAL_LIBS})
		ADD_SUBDIRECTORY (${lib})
	ENDFOREACH (lib)

	FOREACH (lib ${REQUIRED_EXTERNAL_LIBS})
		# add each subdir to a global cached variable for external include directories
		SET (ENV{EXTERNAL_INCLUDE_DIRS} "$ENV{EXTERNAL_INCLUDE_DIRS};${${lib}_SOURCE_DIR}")
		SET (ENV{EXTERNAL_LIB_DEPENDENCIES} "$ENV{EXTERNAL_LIB_DEPENDENCIES};${lib}")
		MESSAGE(STATUS "ADDED ${lib} SOURCE TO VAR: ENV{EXTERNAL_INCLUDE_DIRS} ")
	ENDFOREACH(lib)


    if (USE_CUDA) 
		ADD_SUBDIRECTORY (cuda_libraries)
    endif()

    if (USE_OPENCL) 
        ADD_SUBDIRECTORY (opencl_libraries)
    endif()
