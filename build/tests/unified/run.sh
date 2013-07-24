#!/bin/bash -x


#### Add job submit details here

# DONE: Read grid and generate stencils
# DONE: decompose into P partitions (using metis)
# DONE: Compute weights
# DONE:	read metis output
# DONE:	read stencils
# DONE:	compute weights
# DONE:	write weights
# TODO:	assemble GPU
# TODO:	solve GPU


# -c : number of cols in the grid file (3:X,Y,Z; 4:X,Y,Z,ignore)
# -N : total number of grid points 
# -n : stencil size
# -w : neighbor query method (0:LSH, 1:KDTree, 2:BruteForce)
# -l : LSH grid size (<val>^3 overlaid grid)


MY_MPI_EXE="mpirun -l -perhost 1"
date

MD=165
#N=100000
N=1000
STEN_SIZE=101

NPROC=3
#NPROC=1
#~/sphere_grids/md${MD}.${N}

METIS_FILE=metis_stencils.graph.part.${NPROC}
JOB_RAN_FILE=job_ran

TEST_TYPE=reg
NEW_WORKDIR=./${TEST_TYPE}_${N}_${STEN_SIZE}_${NPROC}proc

GRID_FILE=$HOME/GRIDS/regular/100_cubed/regulargrid_100x_100y_100z_final.ascii

mkdir -p $NEW_WORKDIR
cd $NEW_WORKDIR

if [ ! -f $JOB_RAN_FILE ] 
then

	# Generate a regular grid of size 100^3 
	#./gen_regular_grid.x -x 100 -y 100 -z 100

	# Read grid, generate stencils (Note: -c 4 is required because MD nodes have 4 cols)
	../sten_gen.x -g ${GRID_FILE} -c 3 -N ${N} -n ${STEN_SIZE} -w 0 -l 100

	echo "sten_gen Exit status: $?"

	# Decompose domain for parallel run into NPROC partitions
	gpmetis metis_stencils.graph ${NPROC}

	echo "gpmetis Exit status: $?"

	#	${MY_MPI_EXE} -np ${NPROC} hostname
	# echo "hostname Exit status: $?"

	# add -l for verbose logging of output 
	${MY_MPI_EXE} -np ${NPROC} ../compute_weights.x -w 15 -g input_grid.ascii -a -N ${N} -n ${STEN_SIZE} --eps_c1 0.035 --eps_c2 0.1 -p metis_stencils.graph.part.${NPROC}

	echo "compute_weights Exit status: $?"
	touch $JOB_RAN_FILE
fi

${MY_MPI_EXE} -np ${NPROC} ../evaluate_derivatives_overlap.x -g input_grid.ascii -a -N ${N}  -n ${STEN_SIZE} --eps_c1=0.035 --eps_c2=0.1 -w 15 

echo "evaluate_derivatives Exit status: $?"


#echo "demonstrating 2D regular grid: " 

#./sten_gen.x -g regulargrid_10x_10y_1z_final.ascii -N 100 -n 20 -c 3
#./compute_weights.x -g input_grid.ascii -N 100 -n 20 --eps_c1 0 --eps_c2 0.3 -a -D 2
#./evaluate_derivatives.x -g input_grid.ascii -N 10 -n 20 --eps_c1=0.035 --eps_c2=0.1 -w 15 -a -D 2


#mpirun -output-filename runlog -np 2 ./compute_weights.x -g input_grid.ascii -c 4 -N 4096 -n 17 --eps_c1 0.035 --eps_c2 0.1 -p lsh_100_stencils.graph.part.2

date
