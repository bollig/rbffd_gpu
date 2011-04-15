# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (heat_regulargrid_2d)
add_definitions("-openmp")


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
require_config(test_28x27_works.conf)
require_config(test_28x28_works.conf)
require_config(test_nan_issue.conf)

require_framework_vtk()

IF(VTK_FOUND) 
	ADD_DEFINITIONS("-DUSE_VTK=1")
ENDIF(VTK_FOUND)

# Add one or more serial or parallel tests with arg list
# [executableName] [sourceList] [argvForTest] 
add_serial_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" )
add_parallel_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" ${NUM_PROCS} )
