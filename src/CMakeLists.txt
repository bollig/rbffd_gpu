###############################################
# Project sources
###############################################
	SET(REQUIRED_SOURCE_FILES 
                projectsettings.cpp

		octree.cpp
		parametric_patch.cpp
		exact_ellipsoid.cpp
		exact_ellipse.cpp

		exact_ncar_poisson1.cpp
		exact_ncar_poisson2.cpp

		communicator.cpp
		mpisendable.cpp
		contour_svd.cpp 
		cvt.cpp
                constrained_cvt.cpp
                nested_sphere_cvt.cpp
                ellipse_cvt.cpp
                ellipsoid_cvt.cpp
		derivative.cpp
		derivative_tests.cpp
		gpu.cpp 
		grid.cpp 
		heat.cpp 
		ncar_poisson1.cpp
		rbf.cpp 
		stencils.cpp 
		regulargrid3d.cpp
		regulargrid2d.cpp
		exact_regulargrid.cpp
	)


        SET( REQUIRED_CU_SOURCE_FILES
                ncar_poisson1_cusp.cu
	)

        SET( REQUIRED_CL_SOURCE_FILES
                ncar_poisson1_cl.cpp
		ncar_poisson2_cl.cpp 	# Uses Gordon's change of variable
	)

###############################################
# Project Binaries
###############################################

        # This is a good feature: name your library with "project( <name> )" and CMake will
        # define the variable "${<name>_SOURCE_DIR}" so you can easily access the path
        INCLUDE_DIRECTORIES(
                ${FRAMEWORK_DEP_INCLUDE_DIRS}
        )

	# BUILD A CPU ONLY FRAMEWORK LIBRARY
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
	
IF(CUDA_FOUND)
        MESSAGE(STATUS "ADDING ${FRAMEWORK_CUDA_LIBRARY} TO BUILD PROCESS")

        # BUILD A CUDA ALTERNATIVES LIBRARY
        #CUDA_INCLUDE_DIRECTORIES(${FRAMEWORK_DEP_INCLUDE_DIRS})
        CUDA_ADD_LIBRARY( ${FRAMEWORK_CUDA_LIBRARY} ${REQUIRED_CU_SOURCE_FILES})
        ADD_DEPENDENCIES( ${FRAMEWORK_CUDA_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${FRAMEWORK_LIBRARY})
        #TARGET_LINK_LIBRARIES( ${FRAMEWORK_CUDA_LIBRARY}
        #        ${FRAMEWORK_DEPENDENCIES}
        #        ${amira}
        #        ${armadillo}
        #        ${clapack}
        #        ${fftw}
        #        ${ADDITIONAL_REQUIRED_LIBRARIES} 	# From OS specific CMakeLists.txt
        #)
ENDIF(CUDA_FOUND)

IF(OPENCL_FOUND) 
	MESSAGE(STATUS "ADDING ${FRAMEWORK_OPENCL_LIBRARY} TO BUILD PROCESS")
	ADD_LIBRARY (${FRAMEWORK_OPENCL_LIBRARY} ${REQUIRED_CL_SOURCE_FILES})
	ADD_DEPENDENCIES( ${FRAMEWORK_OPENCL_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${FRAMEWORK_LIBRARY})
	TARGET_LINK_LIBRARIES (${FRAMEWORK_OPENCL_LIBRARY}
			${OPENCL_LIBRARIES}
		)
ENDIF(OPENCL_FOUND)
	

	IF (MPI_FOUND AND USE_MPI)
		TARGET_LINK_LIBRARIES (${FRAMEWORK_LIBRARY}
			${MPI_LIBRARIES}
		)
            IF (CUDA_FOUND)
                TARGET_LINK_LIBRARIES (${FRAMEWORK_CUDA_LIBRARY}
                    ${MPI_LIBRARIES}
                )
            ENDIF (CUDA_FOUND)
	ENDIF (MPI_FOUND AND USE_MPI)
