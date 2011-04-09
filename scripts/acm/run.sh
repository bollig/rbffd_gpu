#$ -cwd
#$ -V
#
# Send email on submission and completion: 
#$ -m be 
#$ -M efb06@fsu.edu
#  -N poster_heat
#$ -o output.$JOB_ID
#
# Merge stdout and stderr?: [y]es [n]o
#$ -j y
#
# Specify the GPU environment (similar to choosing a GPU queue, but
# queue here is interpreted as an execution queue on a single machine.
# The environment is a collection of machines and their queues.)
# 
# 	NOTE: rr = round robin; fu = fill up
# To see -pe options: qconf -spl
#
# -pe make_gts_rr 4
#
# Specify the number of processes we want: 
# 
#

# Binary to run:  
EXEC=~/RBF.framework/build/tests/experiments/heat_regulargrid_2d/heat_regulargrid_2d.x
#EXEC="./test_fftw.sh"


# ARGV for the executable 
EXEC_OPTS="-c test_1024x1024_gpu.conf -o gpu_output_${NSLOTS} -d 1024x1024_${NSLOTS}proc"

# Number of processes per node: 
PPN=1

date
echo ""
echo "HELLO!"
echo "  This job: $JOB_ID, was submitted to SGE using this node for launch: $HOSTNAME"
echo "" 
echo "" 

# for now we dont care if cpus share GPU: 
mpirun -n $NSLOTS ${EXEC} ${EXEC_OPTS}
#mpirun -n $NSLOTS --npernode ${PPN} ${EXEC} ${EXEC_OPTS}
STAT=$?

echo "" 
echo "======================================================="
echo "" 
echo "mpirun exited with status: $STAT"
if [ $STAT == 0 ]; then
	echo " "
	echo "  Normal end of execution."
	echo " "
	date
else 
	echo " "
	echo "  ERROR! Abnormal termination."
	echo " "
	date
	exit $STAT
fi 
echo "======================================================="
