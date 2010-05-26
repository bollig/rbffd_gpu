
###############################################
# External libraries we want to build
###############################################
	SET (EXTERNAL_LIBS
		graphics_lib
	)

	FOREACH (lib ${EXTERNAL_LIBS})
		ADD_SUBDIRECTORY (${lib})
	ENDFOREACH (lib)
