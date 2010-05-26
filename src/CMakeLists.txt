###############################################
# Project sources
###############################################
	SET(REQUIRED_SOURCE_FILES 
		octree.cpp
		parametric_patch.cpp
		exact_ellipsoid.cpp
		communicator.cpp
		mpisendable.cpp
		contour_svd.cpp 
		cvt.cpp
		derivative.cpp
		gpu.cpp 
		grid.cpp 
		#grid3d.cpp
		heat.cpp 
		rbf.cpp 
		stencils.cpp 
		regulargrid3d.cpp
		regulargrid2D.cpp
		exact_regulargrid.cpp
	)


###############################################
# Project Binaries
###############################################
	# BUILD A FRAMEWORK LIBRARY
	ADD_LIBRARY (${FRAMEWORK_LIBRARY} ${REQUIRED_SOURCE_FILES})
	ADD_DEPENDENCIES (${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})

	# Libraries required for the framework (NOT the tests specifically)
	# NOTE: order of libraries is important in Linux. 
	# 	does not matter on macOSX
	TARGET_LINK_LIBRARIES ( ${FRAMEWORK_LIBRARY}
		${FRAMEWORK_DEPENDENCIES}
	  	${amira}
		${armadillo}
		${clapack}
		${fftw}
		${ADDITIONAL_REQUIRED_LIBRARIES} 	# From OS specific CMakeLists.txt
	)

	# This is a good feature: name your library with "project( <name> )" and CMake will 
	# define the variable "${<name>_SOURCE_DIR}" so you can easily access the path
	INCLUDE_DIRECTORIES(
		${FRAMEWORK_DEP_INCLUDE_DIRS}
	)
	
	IF (MPI_FOUND AND USE_MPI)
		TARGET_LINK_LIBRARIES (${FRAMEWORK_LIBRARY}
			${MPI_LIBRARIES}
		)
	ENDIF (MPI_FOUND AND USE_MPI)
