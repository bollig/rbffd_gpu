#!/bin/sh
#PBS -N vortex_roll
#PBS -j oe
#PBS -q batch
#PBS -A UT-NTNL0051

### (USED) Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##

#PBS -l walltime=02:00:00
## THIS IS FOUR PROCS TOTAL (i.e., 2x2): 
#PBS -l nodes=1:ppn=1
##PBS -l feature=cudaworkshop 

### End of PBS options ###

date
#cd $PBS_O_WORKDIR

echo "nodefile="
#cat $PBS_NODEFILE
echo "=end nodefile"

# run the program

which mpirun
pwd 
# mca option binds processes to lowest numbered processor and locks it in until
# termination (i.e. no swtiching)
#mpirun --mca mpi_paffinity_alone 1 ./vortex_rollup_on_sphere.x -d $1 -c $2
./vortex_rollup_on_sphere.x -d $1 -c $2 -o runlog

date

# eof
