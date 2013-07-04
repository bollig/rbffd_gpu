
# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (gpu_one_deriv_one_fun)
#cmake_minimum_required(VERSION 2.6)

# Provided in RBF.framework/common directory 
include(FrameworkMacros)

SET (EXEC_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)

# 3) Source files to compile
SET ( EXEC_SOURCE_FILES
    main.cpp
)

ADD_DEFINITIONS(-g)

# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual
# arguments in quotes ("")) 
SET (ARGV
#   "--verbose"
#   "--file=out.txt"
)

require_config()   # Default is to require test.conf

# Add one or more serial or parallel tests with arg list
# [executableName] [sourceList] [argvForTest] 

add_serial_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" )
LINK_COMMAND( "${rbffd_gpu_SOURCE_DIR}/src/rbffd" "${CMAKE_CURRENT_BINARY_DIR}" "r"  )
LINK_COMMAND( "${rbffd_gpu_SOURCE_DIR}/src/utils"  "${CMAKE_CURRENT_BINARY_DIR}" "u"  )
LINK_COMMAND( "${rbffd_gpu_SOURCE_DIR}/src/cl_kernels" "${CMAKE_CURRENT_BINARY_DIR}" "cl"  )
LINK_COMMAND( "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/src" "src"  )
