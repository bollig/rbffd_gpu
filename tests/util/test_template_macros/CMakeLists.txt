# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (test_template_macros)

# Provided in RBF.framework/common directory 
include(FrameworkMacros)

SET (EXEC_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)

# 3) Source files to compile
SET ( EXEC_SOURCE_FILES
	main.cpp
)

# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual arguments in quotes ("")) 
SET (ARGV
#	"--verbose"
# 	"--file=out.txt"
)

add_serial_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "--file=runlog.txt" )
add_serial_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "--verbose" )
add_parallel_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${argv}" ${NUM_PROCS})

# TODO
#add_serial_cuda_framework_test( ${EXEC_NAME} ${EXEC_SOURCE_FILES} ${argv} )
#add_parallel_cuda_framework_test( ${EXEC_NAME} ${EXEC_SOURCE_FILES} ${argv} ${NUM_PROCS})
#add_serial_opencl_framework_test( ${EXEC_NAME} ${EXEC_SOURCE_FILES} ${argv} )
#add_parallel_opencl_framework_test( ${EXEC_NAME} ${EXEC_SOURCE_FILES} ${argv} ${NUM_PROCS})

require_config() 		# default is to require test.conf
require_config("test2.conf")  	# or we can specify a config file
