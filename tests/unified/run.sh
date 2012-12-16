#!/bin/bash


#### Add job submit details here

./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 0 -l 100
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii lsh_100_stencils.ascii

./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 0 -l 50
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii lsh_050_stencils.ascii

./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 1 
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii kdtree_stencils.ascii

#metis <args> 

#mpirun --hostfile hostfile.4 -np 2 ./runpde 
