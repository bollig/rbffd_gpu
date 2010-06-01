
###############################################
# Graphics_lib libraries we want to build
###############################################
	SET (GRAPHICS_LIB_PROJECTS
		utilities
                timing
	)

	FOREACH (lib ${GRAPHICS_LIB_PROJECTS})
		ADD_SUBDIRECTORY (${lib})
	ENDFOREACH (lib)
