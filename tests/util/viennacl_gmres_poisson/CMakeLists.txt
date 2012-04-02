# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (viennacl_gmres_poisson)

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
)

# Add one or more serial or parallel tests with arg list
# [executableName] [sourceList] [argvForTest] 
add_serial_opencl_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" )
# This binary will show how GMRES converges when we have a nullspace. It
# assumes our free constant is 0. WIth that assumption we can actually obtain
# a good solution. However, if we add dimensions to the problem (i.e., go to
# the stokes flow problem it will not be able to converge as well. 
add_serial_opencl_framework_test( "${PROJECT_NAME}_nullspace.x" "main_nullspace.cpp" "${ARGV}" )


# Show how to use preconditioners
add_serial_opencl_framework_test( "${PROJECT_NAME}_precond.x" "main_precond.cpp" "${ARGV}" )
# Test my own ILU0. ILU with zero fill-in
add_serial_opencl_framework_test( "${PROJECT_NAME}_ilu0.x" "main_ilu0.cpp" "${ARGV}" )


# Stokes steady state (U,V,W,P). Just doign the assembly and testing 1 node locked. 
add_serial_opencl_framework_test( "${PROJECT_NAME}_stokes.x" "main_stokes.cpp" "${ARGV}" )
add_serial_opencl_framework_test( "${PROJECT_NAME}_stokes2.x" "main_stokes2.cpp" "${ARGV}" )
