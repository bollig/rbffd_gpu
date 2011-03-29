#!/bin/bash
#
#  Purpose:
#
#    hello.csh is a sample SGE job file.
#
# Modified:
#
#    06 March 2007
#
#  Author:
#
#    John Burkardt
#
#$ -cwd
#$ -V
# -notify
# Send email on submission and completion: 
#$ -m be 
#$ -M efb06@fsu.edu
#$ -N heat_gpu
#
# Merge stdout and stderr?: [y]es [n]o
#$ -j y
#$ -S /bin/bash
#
# Specify the GPU environment (similar to choosing a GPU queue, but
# queue here is interpreted as an execution queue on a single machine.
# The environment is a collection of machines and their queues.)
# 
# 	NOTE: rr = round robin; fu = fill up
# To see -pe options: qconf -spl
#
#$ -pe make_gts_rr 6
#
# Specify the number of processes we want: 
# 
#

# Binary to run:  
EXEC=../RBF.framework/build/tests/experiments/heat_regulargrid_2d/heat_regulargrid_2d.x

# ARGV for the executable 
EXEC_OPTS="-c ../RBF.framework/build/tests/experiments/heat_regulargrid_2d/test_28x28_works.conf -o runlog -d 28x28"

# Number of processes per node: 
PPN=1

date
echo ""
echo "HELLO!"
echo "  This job was submitted to SGE."
echo "" 
echo `hostname`
echo "" 

mpirun -n $NSLOTS --npernode ${PPN} ${EXEC} ${EXEC_OPTS}

echo " "
echo "HELLO:"
echo "  Normal end of execution."
echo " "
date
