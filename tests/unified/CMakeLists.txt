# 1) Change the PROJECT (<NAME>) to your test name
PROJECT (unified)

# Provided in RBF.framework/common directory 
include(FrameworkMacros)

SET (EXEC_NAME ${PROJECT_NAME}.x)

# 2) Change the number of processors for execution
SET (NUM_PROCS 4)

# 3) Source files to compile
SET ( EXEC_SOURCE_FILES
	main.cu
)

# 4) Runtime arguments for executable (argv[0] .. argv[N]; individual arguments in quotes ("")) 
SET (ARGV
#	"--verbose"
# 	"--file=out.txt"
)

require_config()   # Default is to require test.conf
require_file(run.sh) 
require_file(generate.sh) 
require_file(itasca_job.pbs) 

add_serial_framework_test( "gen_reg_grid.x" "generate_regular_grid.cxx" "${ARGV}" )
add_serial_framework_test( "sten_gen.x" "stencil_generator.cxx" "${ARGV}" )
load_vtk()
add_parallel_framework_test( "compute_weights.x" "compute_weights.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives.x" "evaluate_derivatives.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_nocopyout.x"  "evaluate_derivatives_nocopyout.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_nodecode.x" "evaluate_derivatives_isend_nocopyout.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_prerecv.x" "evaluate_derivatives_isend_prerecv.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_overlap_cpu.x"  "evaluate_derivatives_isend_overlap.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_nowaitencode.x"  "evaluate_derivatives_nowaitencode.cxx" "${ARGV}" 1)
add_parallel_opencl_framework_test( "evaluate_derivatives_overlap.x" "evaluate_derivatives_overlap.cxx" "${ARGV}" 1)
add_parallel_opencl_framework_test( "evaluate_derivatives_nooverlap.x" "evaluate_derivatives_nooverlap.cxx" "${ARGV}" 1)
add_parallel_opencl_framework_test( "evaluate_derivatives_overlap_fast.x" "evaluate_derivatives_overlap_fast.cxx" "${ARGV}" 1)
add_parallel_opencl_framework_test( "evaluate_derivatives_overlap_faster.x" "evaluate_derivatives_overlap_faster.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_isend.x" "evaluate_derivatives_isend.cxx" "${ARGV}" 1)
add_parallel_framework_test( "evaluate_derivatives_alltoallv.x" "evaluate_derivatives_alltoallv.cxx" "${ARGV}" 1)

# Add one or more serial or parallel tests with arg list
# [executableName] [sourceList] [argvForTest] 
#add_serial_cuda_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" )
#add_parallel_cuda_framework_test( "${EXEC_NAME}" "${EXEC_SOURCE_FILES}" "${ARGV}" ${NUM_PROCS} )
