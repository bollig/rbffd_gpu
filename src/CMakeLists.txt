###############################################
# Project sources
###############################################
SET(REQUIRED_SOURCE_FILES 
    # find * -name "*.c*" |grep -v "\.cu" |grep -v "cl\.c"
    exact_solutions/exact_ellipse.cpp
    exact_solutions/exact_ellipsoid.cpp
    exact_solutions/exact_ncar_poisson1.cpp
    exact_solutions/exact_ncar_poisson2.cpp
    exact_solutions/exact_regulargrid.cpp
    exact_solutions/exact_regulargrid_nonuniform.cpp
    #TODO    exact_solutions/exact_regulargrid_convection.cpp
    exact_solutions/exact_regulargrid_shu2006.cpp

    grids/cvt/cvt.cpp
    #grids/cvt/constrained_cvt.cpp
    grids/cvt/ellipse_cvt.cpp
    grids/cvt/ellipsoid_cvt.cpp
    grids/cvt/alt_ellipsoid_cvt.cpp
    grids/cvt/nested_ellipse_cvt.cpp
    grids/regulargrid.cpp
    grids/grid_reader.cpp
    grids/grid_interface.cpp
    grids/domain.cpp
    grids/domain_nompi.cpp
    grids/ge_rcm_single.cpp


    #    pdes/parabolic/heat.cpp
    pdes/pde.cpp
    pdes/time_dependent_pde.cpp
    pdes/parabolic/heat_pde.cpp
    #TODO    pdes/parabolic/convection_diffusion_pde.cpp
    pdes/elliptic/ncar_poisson1.cpp
    pdes/elliptic/stokes_steady_pde.cpp
    pdes/hyperbolic/vortex_rollup.cpp
    pdes/hyperbolic/cosine_bell.cpp

    rbffd/rbffd.cpp
    rbffd/derivative_tests.cpp
    rbffd/rbfs/rbf.cpp

    # Need to cleanup: 
    rbffd/contour_svd.cpp
    rbffd/stencils.cpp

    utils/comm/communicator.cpp
    utils/comm/mpisendable.cpp
    utils/conf/projectsettings.cpp
    utils/geom/parametric_patch.cpp
    utils/geom/octree.cpp	
    utils/geom/cart2sph.cpp
    utils/random.cpp
    utils/norms.cpp
    )


SET( REQUIRED_CU_SOURCE_FILES
    # find * -name "*.cu"
    #pdes/elliptic/ncar_poisson1_cusp.cu
    pdes/parabolic/heat_pde_cusp.cu
    )

SET( REQUIRED_CL_SOURCE_FILES
    utils/opencl/cl_base_class.cpp

    # find * -name "*cl.c*"
    #*	pdes/elliptic/ncar_poisson1_cl.cpp
    #P1 pdes/elliptic/ncar_poisson2_cl.cpp
    #REQUIRES BOOST:    
    #TODO: Update    pdes/elliptic/nonuniform_poisson1_cl.cpp
    pdes/parabolic/heat_pde_cl.cpp
    #    pdes/hyperbolic/cosine_bell_cl.cpp
    pdes/time_dependent_pde_cl.cpp
    pdes/time_dependent_pde_vcl.cpp
    rbffd/rbffd_cl.cpp
    rbffd/fun_cl.cpp
    #rbffd/fun_vcl.cpp
    rbffd/rbffd_single_cl.cpp
    rbffd/rbffd_multi_cl.cpp
	rbffd/rbffd_multi_weight_cl.cpp
	rbffd/rbffd_multi_weight_fun_cl.cpp
	rbffd/rbffd_multi_weight_fun4_cl.cpp
    rbffd/rbffd_vcl.cpp
    ../external/opencl_libraries/include/pugixml/src/pugixml.cpp
)

###############################################
# Project Binaries
###############################################

# This is a good feature: name your library with "project( <name> )" and CMake will
# define the variable "${<name>_SOURCE_DIR}" so you can easily access the path
INCLUDE_DIRECTORIES(
    ${FRAMEWORK_DEP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
	${/mnt/global/LCSE/gerlebacher/rbf/external/}
    )

# BUILD A CPU ONLY FRAMEWORK LIBRARY
# noop.c is an empty file because OSX requires at least one source file
# when calling "ar" and we cant guarantee the list will not be empty
ADD_LIBRARY (${FRAMEWORK_LIBRARY} ${REQUIRED_SOURCE_FILES} noop.cxx)
ADD_DEPENDENCIES (${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})

# Libraries required for the framework (NOT the tests specifically)
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
TARGET_LINK_LIBRARIES ( ${FRAMEWORK_LIBRARY}
    ${FRAMEWORK_DEPENDENCIES}
    ${amira}
    ${armadillo}
    ${clapack}
    ${fftw3}
    ${ADDITIONAL_REQUIRED_LIBRARIES} 	# From OS specific CMakeLists.txt
    )

IF(CUDA_FOUND AND USE_CUDA )
    MESSAGE(STATUS "ADDING ${FRAMEWORK_CUDA_LIBRARY} TO BUILD PROCESS")

    # BUILD A CUDA ALTERNATIVES LIBRARY
    #CUDA_INCLUDE_DIRECTORIES(${FRAMEWORK_DEP_INCLUDE_DIRS})
    # noop.c is an empty file because OSX requires at least one file
    # when calling "ar" and we cant guarantee the list will not be empty
    CUDA_ADD_LIBRARY( ${FRAMEWORK_CUDA_LIBRARY} ${REQUIRED_CU_SOURCE_FILES} noop.cxx)
    ADD_DEPENDENCIES( ${FRAMEWORK_CUDA_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${FRAMEWORK_LIBRARY})
    #TARGET_LINK_LIBRARIES( ${FRAMEWORK_CUDA_LIBRARY}
    #        ${FRAMEWORK_DEPENDENCIES}
    #        ${amira}
    #        ${armadillo}
    #        ${clapack}
    #        ${fftw3}
    #        ${ADDITIONAL_REQUIRED_LIBRARIES} 	# From OS specific CMakeLists.txt
    #)
ENDIF(CUDA_FOUND AND USE_CUDA)

IF(OPENCL_FOUND AND USE_OPENCL) 
    MESSAGE(STATUS "ADDING ${FRAMEWORK_OPENCL_LIBRARY} TO BUILD PROCESS ${OPENCL_LIBRARIES}")
    INCLUDE_DIRECTORIES( 
        ${OPENCL_INCLUDE_DIR}
        )
    ADD_LIBRARY (${FRAMEWORK_OPENCL_LIBRARY} ${REQUIRED_CL_SOURCE_FILES} noop.cxx)
    ADD_DEPENDENCIES( ${FRAMEWORK_OPENCL_LIBRARY} ${FRAMEWORK_DEPENDENCIES} ${FRAMEWORK_LIBRARY})
    TARGET_LINK_LIBRARIES (	${FRAMEWORK_OPENCL_LIBRARY}
        ${OPENCL_LIBRARIES}
        )
ENDIF(OPENCL_FOUND AND USE_OPENCL)


IF (MPI_FOUND AND USE_MPI)
    TARGET_LINK_LIBRARIES (	${FRAMEWORK_LIBRARY}
        ${MPI_LIBRARIES}
        )
    IF (CUDA_FOUND)
        TARGET_LINK_LIBRARIES (	${FRAMEWORK_CUDA_LIBRARY}
            ${MPI_LIBRARIES}
            )
    ENDIF (CUDA_FOUND)
ENDIF (MPI_FOUND AND USE_MPI)
