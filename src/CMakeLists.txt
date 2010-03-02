###############################################
# Project sources
###############################################
	SET(REQUIRED_SOURCE_FILES 
		communicator.cpp
		mpisendable.cpp
		contour_svd.cpp 
		cvt.cpp
		derivative.cpp
		gpu.cpp 
		grid.cpp 
		heat.cpp 
		rbf.cpp 
		stencils.cpp 
	)

###############################################
# Project Binaries
###############################################
	# BUILD A FRAMEWORK LIBRARY
	ADD_LIBRARY (${FRAMEWORK_LIBRARY} ${REQUIRED_SOURCE_FILES})

	# Libraries required for the framework (NOT the tests specifically)
	# NOTE: order of libraries is important in Linux. 
	# 	does not matter on macOSX
	TARGET_LINK_LIBRARIES ( ${FRAMEWORK_LIBRARY} 
		${utilities}
	  	${amira}
		${armadillo}
		${clapack}
		${fftw}
		${ADDITIONAL_REQUIRED_LIBRARIES} 	# From OS specific CMakeLists.txt
	)
	
	IF (MPI_FOUND AND USE_MPI)
		TARGET_LINK_LIBRARIES (${FRAMEWORK_LIBRARY}
			${MPI_LIBRARIES}
		)
	ENDIF (MPI_FOUND AND USE_MPI)
