# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (viennacl_stokes_pgmres)

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

require_config()   # Default is to require test.conf
require_config(test_N20.conf)   # Default is to require test.conf

load_vtk()

# Add one or more serial or parallel tests with arg list
# [executableName] [sourceList] [argvForTest] 
add_serial_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" )
add_parallel_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" ${NUM_PROCS} )
