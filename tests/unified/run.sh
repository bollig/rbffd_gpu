#!/bin/bash


#### Add job submit details here

# DONE: Read grid and generate stencils
# DONE: decompose into P partitions (using metis)
# DONE: Compute weights
# DONE:	read metis output
# DONE:	read stencils
# DONE:	compute weights
# DONE:	write weights
# TODO:	assemble 
# TODO:	solve


# -c : number of cols in the grid file (3:X,Y,Z; 4:X,Y,Z,ignore)
# -N : total number of grid points 
# -n : stencil size
# -w : neighbor query method (0:LSH, 1:KDTree, 2:BruteForce)
# -l : LSH grid size (<val>^3 overlaid grid)

export NPROC=16

# Read grid, generate stencils (Note: -c 4 is required because MD nodes have 4 cols)
./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 0 -l 100

# Decompose domain for parallel run into NPROC partitions
gpmetis metis_stencils.graph ${NPROC}

# Compute weights for each partition (WARNING: may produce large output files)
echo "STARTING WEIGHTS"
mpirun -np ${NPROC} ./compute_weights.x -g input_grid.ascii -a -N 4096 -n 17 --eps_c1 0.035 --eps_c2 0.1 -p metis_stencils.graph.part.${NPROC}

#mpirun -output-filename runlog -np 2 ./compute_weights.x -g input_grid.ascii -c 4 -N 4096 -n 17 --eps_c1 0.035 --eps_c2 0.1 -p lsh_100_stencils.graph.part.2

