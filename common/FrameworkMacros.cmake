#SET ( TEST_COUNT 0 ) 

############################################
# 	REQUIRE CONFIG
# If a config file is required it is copied 
# into the binary directory. 
# NOTE: two options here allow us to use a 
# default config name or to specify one 
############################################
MACRO ( REQUIRE_CONFIG )
	if ( ${ARGC} LESS 1 )
		set (_configfilename ${CMAKE_CURRENT_SOURCE_DIR}/test.conf)
	else (${ARGC} LESS 1)
		set (_configfilename ${CMAKE_CURRENT_SOURCE_DIR}/${ARGV0})
	endif (${ARGC} LESS 1)
	if ( EXISTS ${_configfilename} )	
		# COPY ANY CONFIG FILES REQUIRED BY THE TEST
		FILE(COPY ${_configfilename} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	else (EXISTS ${_configfilename})
		MESSAGE(SEND_ERROR "\nERROR! File: ${_configfilename} required for testing does not exist.")
	endif ( EXISTS ${_configfilename} )
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
	ADD_TEST (${_full_test_name} "${_execname}" ${ARGV})
	increment(TEST_COUNT_${_execname})
ENDMACRO ( ADD_SERIAL_FRAMEWORK_TEST )


MACRO ( ADD_PARALLEL_FRAMEWORK_TEST _execname _sourcelist _argv _numprocs )
	SET (_full_test_name "${_execname}_test${TEST_COUNT}") 
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
			ADD_TEST (${_full_test_name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_numprocs} ${MPIEXEC_PREFLAGS} "${_execname}" ${_argv} ${MPIEXEC_POSTFLAGS})
			increment(TEST_COUNT_${_execname})
		ELSE ( USE_MPI )
			MESSAGE (WARNING "\nWARNING! Test ${_full_test_name} is disabled because USE_MPI=FALSE.")
		ENDIF ( USE_MPI )
	ELSE (MPI_FOUND) 
		MESSAGE (WARNING "\nWARNING! Could not add parallel test ${_full_test_name} because MPI was not found.") 
	ENDIF (MPI_FOUND)
ENDMACRO ( ADD_PARALLEL_FRAMEWORK_TEST _execname _argv _numprocs )


