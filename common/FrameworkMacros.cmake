
# ITS NOT CLEAR WHY CMAKE SUDDENLY REQUIRES THIS CRAP.
#cmake_policy(PUSH)
#cmake_minimum_required(VERSION 2.6.3)
#cmake_policy(POP)
#CMAKE_POLICY(SET CMP0012 OLD) 

#SET ( TEST_COUNT 0 ) 

MACRO ( COPY_KERNEL_SOLVER_COMMAND  _source )

    if ( ${ARGC} LESS 2 )
        set (_destination "solver.cl")
    else (${ARGC} LESS 2)
        set (_destination ${ARGV1})
    endif (${ARGC} LESS 2)

    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_destination}
        COMMAND #${CMAKE_COMMAND} copy
        cp ${CMAKE_SOURCE_DIR}/src/cl_kernels/${_source} ${CMAKE_CURRENT_BINARY_DIR}/${_destination}
        DEPENDS ${CMAKE_SOURCE_DIR}/src/cl_kernels/${_source}
    )
    SET (_which "${_source}_to_${_destination}")
    
    IF (NOT DEFINED ${TARGET_COUNT_${_which}})
        SET(TARGET_COUNT_${_which} 0 CACHE TYPE INTERNAL)
        MARK_AS_ADVANCED(TARGET_COUNT_${_which})
    ENDIF (NOT DEFINED ${TARGET_COUNT_${_which}})

    increment(TARGET_COUNT_${_which})
    SET (FULL_TARGET_NAME "Copy ${_source} to ${_destination} ${TARGET_COUNT_${_which}}")
    ADD_CUSTOM_TARGET(${FULL_TARGET_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_destination})

ENDMACRO ( COPY_KERNEL_SOLVER_COMMAND _source _destination)

MACRO ( COPY_FILE_COMMAND _source_filename _source_dir _dest_filename _dest_dir)

    SET (_source "${_source_dir}/${_source_filename}")
    SET (_destination "${_dest_dir}/${_dest_filename}")

    ADD_CUSTOM_COMMAND(
        OUTPUT ${_dest_filename}
        COMMAND 
        cp ${_source} ${_destination}
        DEPENDS ${_source_filename}
    )
    SET (_which "${_source_filename}_to_${_dest_filename}")
    
    IF (NOT DEFINED ${TARGET_COUNT_${_which}})
        SET(TARGET_COUNT_${_which} 0 CACHE TYPE INTERNAL)
        MARK_AS_ADVANCED(TARGET_COUNT_${_which})
    ENDIF (NOT DEFINED ${TARGET_COUNT_${_which}})

    STRING(REPLACE "/" "_" _dest_dir_no_seps ${CMAKE_CURRENT_BINARY_DIR})

    increment(TARGET_COUNT_${_which})
    SET (FULL_TARGET_NAME "Copy ${_source_filename} to ${_dest_dir_no_seps}_${TARGET_COUNT_${_which}}")
    ADD_CUSTOM_TARGET(${FULL_TARGET_NAME} ALL DEPENDS ${_destination})
ENDMACRO ( COPY_FILE_COMMAND _source_filename _source_dir _dest_filename _dest_dir)

MACRO ( LOAD_VTK )
    # if VTK_FOUND and ENABLED
    #   assume the rbf_vtk lib was built
    if (USE_VTK)
        find_package(VTK)
        if (VTK_FOUND)
            INCLUDE( ${USE_VTK_FILE} )
            message( STATUS "VTK found, updating FRAMEWORK_{...} variables.")
            set (FRAMEWORK_DEP_INCLUDE_DIRS ${FRAMEWORK_DEP_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
            # NOTE: we need to fix these libraries in the future
            set (FRAMEWORK_DEPENDENCIES ${FRAMEWORK_DEPENDENCIES} vtkHybrid vtkWidgets)
            ADD_DEFINITIONS("-DUSE_VTK=1")
        else (VTK_FOUND)
            message(WARNING "VTK was not found.")
            ADD_DEFINITIONS("-DUSE_VTK=0")
        endif (VTK_FOUND)
    else (USE_VTK)
        message( STATUS "USE_VTK=off, not loading VTK.")
        ADD_DEFINITIONS("-DUSE_VTK=0")
    endif (USE_VTK)
ENDMACRO( LOAD_VTK )



MACRO ( REQUIRE_PETSC )
    message( STATUS "Searching for PETSc" )
    find_package(PETSc)
    if (PETSC_FOUND)
        message( STATUS "PETSc found, updating FRAMEWORK_{...} variables.")
        #message( STATUS "These include dirs will be added to the search: ${PETSC_INCLUDES}")
        set (FRAMEWORK_DEP_INCLUDE_DIRS ${FRAMEWORK_DEP_INCLUDE_DIRS} ${PETSC_INCLUDES})
        #message( STATUS "These libraries will be added to the link list: ${PETSC_LIBRARIES}")
        set (FRAMEWORK_DEPENDENCIES ${FRAMEWORK_DEPENDENCIES} ${PETSC_LIBRARIES})
        # Might not be necessary: 
        set (MPIEXEC ${PETSC_MPIEXEC})
        add_definitions(${PETSC_DEFINITIONS})
    else (PETSC_FOUND)
        message(ERROR "A PETSc installation is required to proceed.")
    endif (PETSC_FOUND)
ENDMACRO( REQUIRE_PETSC )


MACRO ( PRINT_DEPS )

    message( STATUS "Framework Dependency Includes: ${FRAMEWORK_DEP_INCLUDE_DIRS}")
    message( STATUS "Framework Dependency Libraries: ${FRAMEWORK_DEPENDENCIES}")

ENDMACRO ( PRINT_DEPS )

############################################
# 	REQUIRE CONFIG
# If a config file is required it is copied 
# into the binary directory. 
# NOTE: two options here allow us to use a 
# default config name or to specify one 
############################################
MACRO ( REQUIRE_CONFIG )
    if ( ${ARGC} LESS 1 )
        set (_filename test.conf)
    else (${ARGC} LESS 1)
        set (_filename ${ARGV0})
    endif (${ARGC} LESS 1)
        
    set (_source ${CMAKE_CURRENT_SOURCE_DIR}/${_filename})
    set (_destination ${CMAKE_CURRENT_BINARY_DIR}/${_filename})
    
    if ( EXISTS ${_source} )	
        # COPY ANY CONFIG FILES REQUIRED BY THE TEST
        # FILE(COPY ${_configfilename} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

        COPY_FILE_COMMAND( "${_filename}" "${CMAKE_CURRENT_SOURCE_DIR}" "${_filename}" "${CMAKE_CURRENT_BINARY_DIR}" )

    else (EXISTS ${_source})
        MESSAGE(SEND_ERROR "\nERROR! File: ${_source} required for testing does not exist.")
    endif ( EXISTS ${_source} )
ENDMACRO(REQUIRE_CONFIG)

MACRO ( REQUIRE_FILE _filename)
    require_config(${_filename})
ENDMACRO( REQUIRE_FILE )



#############################################
#   	ADDING TESTS 
# Simplify the task of adding a test case for 
# our experiments
#############################################


# SIMPLE INCREMENT BORROWED FROM http://mima2.jouy.inra.fr/git/contrib-itk/WrapITK/CMakeUtilityFunctions.cmake
MACRO(INCREMENT var_name)
    # Increment the input variable (must be in [empty;0,9]) and store the result in var_name.
    SET(${var_name} ${inc${${var_name}}})
    IF(NOT DEFINED ${var_name})
        MESSAGE(FATAL_ERROR "Could not increment ${var_name}. Input out of range 0-9?")
    ENDIF(NOT DEFINED ${var_name})
ENDMACRO(INCREMENT)

# Added (inc 1) which allows us to increment a variable that does not exist
SET(inc 1)
SET(inc0 1)
SET(inc1 2)
SET(inc2 3)
SET(inc3 4)
SET(inc4 5)
SET(inc5 6)
SET(inc6 7)
SET(inc7 8)
SET(inc8 9)
# allow up to 10 tests 
SET(inc9 10)

######### ADD A NEW SERIAL TEST GIVEN NAME AND ARGS ONLY ########
MACRO ( ADD_SERIAL_FRAMEWORK_TEST _execname _sourcelist _argv)
    # Make sure the RBF.framework library is built and linked to this test
    INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
    IF (NOT DEFINED ${TEST_COUNT_${_execname}})
        SET(TEST_COUNT_${_execname} 0 CACHE TYPE INTERNAL)
        MARK_AS_ADVANCED(TEST_COUNT_${_execname})
    ENDIF (NOT DEFINED ${TEST_COUNT_${_execname}})

    IF (${TEST_COUNT_${_execname}} MATCHES 0) 
        ADD_EXECUTABLE(${_execname} ${_sourcelist})
    ENDIF (${TEST_COUNT_${_execname}} MATCHES 0) 


    SET (_full_test_name "${_execname}_serial_test${TEST_COUNT_${_execname}}") 

    ADD_DEPENDENCIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})
    TARGET_LINK_LIBRARIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES}) 
    # Number tests according 
    MESSAGE(STATUS "ADDING TEST: ${_full_test_name}" )
    ADD_TEST (NAME ${_full_test_name} COMMAND "${_execname}" ${_argv})
    increment(TEST_COUNT_${_execname})
ENDMACRO ( ADD_SERIAL_FRAMEWORK_TEST _execname _sourcelist _argv)


MACRO ( ADD_PARALLEL_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs )
    SET (_full_test_name "${_execname}_parallel_test") 
    IF (MPI_FOUND)
        IF ( USE_MPI)
            # Make sure the RBF.framework library is built and linked to this test
            INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
            IF (NOT DEFINED ${TEST_COUNT_${_execname}})
                SET(TEST_COUNT_${_execname} 0 CACHE TYPE INTERNAL)
                MARK_AS_ADVANCED(TEST_COUNT_${_execname})
            ENDIF (NOT DEFINED ${TEST_COUNT_${_execname}})

            IF (${TEST_COUNT_${_execname}} MATCHES 0) 
                ADD_EXECUTABLE(${_execname} ${_sourcelist})
            ENDIF (${TEST_COUNT_${_execname}} MATCHES 0) 

            SET (_full_test_name "${_execname}_parallel_test${TEST_COUNT_${_execname}}") 

            # Make doubly sure the mpi libs are linked into the executable
            TARGET_LINK_LIBRARIES (${_execname}
                ${MPI_LIBRARIES}
                )

            # Add a Parallel Test
            # Format: ADD_TEST( [TestName] [MPIExecutable] [MPINumProcFlag] [#ofProcs] [MPIOptions] [Executable] [Arg1] [Arg2] ... [ArgN])
            MESSAGE(STATUS "ADDING TEST: ${_full_test_name}" )
            ADD_TEST (NAME ${_full_test_name} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_numprocs} ${MPIEXEC_PREFLAGS} "${_execname}" ${_argv} ${MPIEXEC_POSTFLAGS})
            increment(TEST_COUNT_${_execname})
        ELSE ( USE_MPI )
            MESSAGE (WARNING "\nWARNING! Test ${_full_test_name} is disabled because USE_MPI=FALSE.")
        ENDIF ( USE_MPI )
    ELSE (MPI_FOUND) 
        MESSAGE (WARNING "\nWARNING! Could not add parallel test ${_full_test_name} because MPI was not found.") 
    ENDIF (MPI_FOUND)
ENDMACRO ( ADD_PARALLEL_FRAMEWORK_TEST _execname _argv _numprocs )



MACRO ( ADD_SERIAL_OPENCL_FRAMEWORK_TEST _execname _sourcelist _argv)
    SET (_full_test_name "${_execname}_serial_opencl_test${TEST_COUNT}") 
    IF (OPENCL_FOUND)
        ADD_SERIAL_FRAMEWORK_TEST( "${_execname}" "${_sourcelist}" "${_argv}")
        ADD_DEPENDENCIES( ${_execname} ${FRAMEWORK_OPENCL_LIBRARY} )
        TARGET_LINK_LIBRARIES ( ${_execname} ${FRAMEWORK_OPENCL_LIBRARY} )
    ELSE (OPENCL_FOUND)
        MESSAGE (WARNING "\nWARNING! ${_full_test_name} is disabled because OpenCL was not found.")
    ENDIF (OPENCL_FOUND)
ENDMACRO ( ADD_SERIAL_OPENCL_FRAMEWORK_TEST _execname _sourcelist _argv)



MACRO ( ADD_PARALLEL_OPENCL_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs)
    SET (_full_test_name "${_execname}_parallel_opencl_test${TEST_COUNT}") 
    IF (OPENCL_FOUND ) 
        ADD_PARALLEL_FRAMEWORK_TEST( "${_execname}" "${_sourcelist}" "${_argv}" "${_numprocs}")
        ADD_DEPENDENCIES( ${_execname} ${FRAMEWORK_OPENCL_LIBRARY} )
        TARGET_LINK_LIBRARIES ( ${_execname} 
            ${FRAMEWORK_OPENCL_LIBRARY} 
            )
    ELSE (OPENCL_FOUND)
        MESSAGE (WARNING "\nWARNING! ${_full_test_name} is disabled because OpenCL was not found.")
    ENDIF (OPENCL_FOUND)
ENDMACRO (ADD_PARALLEL_OPENCL_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs)

MACRO ( ADD_SERIAL_CUDA_FRAMEWORK_TEST _execname _sourcelist _argv)
    SET (_full_test_name "${_execname}_serial_cuda_test${TEST_COUNT}") 
    IF (CUDA_FOUND )
        # Make sure the RBF.framework library is built and linked to this test
        INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
        CUDA_INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
        IF (NOT DEFINED ${TEST_COUNT_${_execname}})
            SET(TEST_COUNT_${_execname} 0 CACHE TYPE INTERNAL)
            MARK_AS_ADVANCED(TEST_COUNT_${_execname})
        ENDIF (NOT DEFINED ${TEST_COUNT_${_execname}})

        IF (${TEST_COUNT_${_execname}} MATCHES 0) 
            CUDA_ADD_EXECUTABLE (${_execname} ${_sourcelist} )
        ENDIF (${TEST_COUNT_${_execname}} MATCHES 0) 

        SET (_full_test_name "${_execname}_serial_test${TEST_COUNT_${_execname}}") 

        ADD_DEPENDENCIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})
        ADD_DEPENDENCIES( ${_execname} ${FRAMEWORK_CUDA_LIBRARY} )
        TARGET_LINK_LIBRARIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES}) 
        TARGET_LINK_LIBRARIES ( ${_execname} ${FRAMEWORK_CUDA_LIBRARY} )
        # Number tests according 
        MESSAGE(STATUS "ADDING TEST: ${_full_test_name}" )
        ADD_TEST (NAME ${_full_test_name} COMMAND "${_execname}" ${_argv})
        increment(TEST_COUNT_${_execname})
    ELSE (CUDA_FOUND)
        MESSAGE (WARNING "\nWARNING! ${_full_test_name} is disabled because CUDA was not found.")
    ENDIF (CUDA_FOUND)
ENDMACRO (ADD_SERIAL_CUDA_FRAMEWORK_TEST _execname _sourcelist _argv)

MACRO ( ADD_PARALLEL_CUDA_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs)
    SET (_full_test_name "${_execname}_test${TEST_COUNT}") 
    IF (CUDA_FOUND)
        IF (MPI_FOUND)
            IF ( USE_MPI)
                # Make sure the RBF.framework library is built and linked to this test
                INCLUDE_DIRECTORIES(${FRAMEWORK_INCLUDE_DIR} ${FRAMEWORK_DEP_INCLUDE_DIRS})
                IF (NOT DEFINED ${TEST_COUNT_${_execname}})
                    SET(TEST_COUNT_${_execname} 0 CACHE TYPE INTERNAL)
                    MARK_AS_ADVANCED(TEST_COUNT_${_execname})
                ENDIF (NOT DEFINED ${TEST_COUNT_${_execname}})

                IF (${TEST_COUNT_${_execname}} MATCHES 0) 
                    CUDA_ADD_EXECUTABLE (${_execname} ${_sourcelist} )
                ENDIF (${TEST_COUNT_${_execname}} MATCHES 0) 

                SET (_full_test_name "${_execname}_parallel_test${TEST_COUNT_${_execname}}") 

                ADD_DEPENDENCIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES})
                ADD_DEPENDENCIES( ${_execname} ${FRAMEWORK_CUDA_LIBRARY} )
                TARGET_LINK_LIBRARIES (${_execname} ${FRAMEWORK_LIBRARY} ${FRAMEWORK_DEPENDENCIES}) 
                TARGET_LINK_LIBRARIES ( ${_execname} ${FRAMEWORK_CUDA_LIBRARY} )

                # Make doubly sure the mpi libs are linked into the executable
                TARGET_LINK_LIBRARIES (${_execname}
                    ${MPI_LIBRARIES}
                    )

                # Add a Parallel Test
                # Format: ADD_TEST( [TestName] [MPIExecutable] [MPINumProcFlag] [#ofProcs] [MPIOptions] [Executable] [Arg1] [Arg2] ... [ArgN])
                MESSAGE(STATUS "ADDING TEST: ${_full_test_name}" )
                ADD_TEST (NAME ${_full_test_name} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_numprocs} ${MPIEXEC_PREFLAGS} "${_execname}" ${_argv} ${MPIEXEC_POSTFLAGS})
                increment(TEST_COUNT_${_execname})
            ELSE ( USE_MPI )
                MESSAGE (WARNING "\nWARNING! Test ${_full_test_name} is disabled because USE_MPI=FALSE.")
            ENDIF ( USE_MPI )
        ELSE (MPI_FOUND) 
            MESSAGE (WARNING "\nWARNING! Could not add parallel test ${_full_test_name} because MPI was not found.") 
        ENDIF (MPI_FOUND)
    ELSE (CUDA_FOUND)
        MESSAGE (WARNING "\nWARNING! ${_full_test_name} is disabled because CUDA was not found.")
    ENDIF (CUDA_FOUND)
ENDMACRO (ADD_PARALLEL_CUDA_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs)
