#!/bin/bash

# TODO:	assemble 
# TODO:	solve

export NPROC=4
export NSIZE=04096
export STENSIZE=31

# Read grid, generate stencils (Note: -c 4 is required because MD nodes have 4 cols)
../sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N ${NSIZE} -n ${STENSIZE} -w 0 -l 100

# Decompose domain for parallel run into NPROC partitions
gpmetis metis_stencils.graph ${NPROC}

# Compute weights for each partition (WARNING: may produce large output files)
mpirun -np ${NPROC} ../compute_weights.x -w 15 -g input_grid.ascii -a -N ${NSIZE} -n ${STENSIZE} --eps_c1 0.035 --eps_c2 0.1 -p metis_stencils.graph.part.${NPROC}

mpirun -np ${NPROC} ../evaluate_derivatives.x -g input_grid.ascii -a -N ${NSIZE} -n ${STENSIZE} --eps_c1=0.035 --eps_c2=0.1 -w 15

#echo "demonstrating 2D regular grid: " 
#./sten_gen.x -g regulargrid_10x_10y_1z_final.ascii -N 100 -n 20 -c 3
#./compute_weights.x -g input_grid.ascii -N 100 -n 20 --eps_c1 0 --eps_c2 0.3 -a -D 2
#./evaluate_derivatives.x -g input_grid.ascii -N 10 -n 20 --eps_c1=0.035 --eps_c2=0.1 -w 15 -a -D 2

