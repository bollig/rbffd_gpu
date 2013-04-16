#!/bin/bash


#### Add job submit details here

# 1) Read grid and generate stencils
# TODO:	How to handle boundary? 
# 
# 2) decompose into P partitions (using metis)
# 3) Compute weights
# TODO: read grid
# TODO:	read metis output
# TODO:	read stencils
# TODO:	compute weights
# TODO:	write weights
# TODO:	assemble 
# TODO:	solve


# -c : number of cols in the grid file (3:X,Y,Z; 4:X,Y,Z,ignore)
# -N : total number of grid points 
# -n : stencil size
# -w : neighbor query method (0:LSH, 1:KDTree, 2:BruteForce)
# -l : LSH grid size (<val>^3 overlaid grid)

# 4096

./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 0 -l 100
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii lsh_100_stencils.ascii
mv metis_stencils.graph lsh_100_stencils.graph
gpmetis lsh_100_stencils.graph 2
#mpirun -hostfile hostfile.2 -np 2 ./compute_weights.x -g ~/sphere_grids/md063.04096 -s lsh_100_stencils.ascii -p lsh_100_stencils.graph.part.2


./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 0 -l 50
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii lsh_050_stencils.ascii
mv metis_stencils.graph lsh_050_stencils.graph
gpmetis lsh_050_stencils.graph 2


./sten_gen.x -g ~/sphere_grids/md063.04096 -c 4 -N 4096 -n 17 -w 1 
mv stencils_maxsz17_loadedgrid_4096nodes_final.ascii kdtree_stencils.ascii
mv metis_stencils.graph kdtree_stencils.graph
gpmetis kdtree_stencils.graph 2

#
## 10201
#
#./sten_gen.x -g ~/sphere_grids/md100.10201 -c 4 -N 10201 -n 17 -w 0 -l 100
#mv stencils_maxsz17_loadedgrid_10201nodes_final.ascii lsh_100_stencils.ascii
#
#./sten_gen.x -g ~/sphere_grids/md100.10201 -c 4 -N 10201 -n 17 -w 0 -l 50
#mv stencils_maxsz17_loadedgrid_10201nodes_final.ascii lsh_050_stencils.ascii
#
#./sten_gen.x -g ~/sphere_grids/md100.10201 -c 4 -N 10201 -n 17 -w 1 
#mv stencils_maxsz17_loadedgrid_10201nodes_final.ascii kdtree_stencils.ascii
#
## 16384
#
#./sten_gen.x -g ~/sphere_grids/md127.16384 -c 4 -N 16384 -n 17 -w 0 -l 100
#mv stencils_maxsz17_loadedgrid_16384nodes_final.ascii lsh_100_stencils.ascii
#
#./sten_gen.x -g ~/sphere_grids/md127.16384 -c 4 -N 16384 -n 17 -w 0 -l 50
#mv stencils_maxsz17_loadedgrid_16384nodes_final.ascii lsh_050_stencils.ascii
#
#./sten_gen.x -g ~/sphere_grids/md127.16384 -c 4 -N 16384 -n 17 -w 1 
#mv stencils_maxsz17_loadedgrid_16384nodes_final.ascii kdtree_stencils.ascii
#

#metis <args> 

#mpirun --hostfile hostfile.4 -np 2 ./runpde 
