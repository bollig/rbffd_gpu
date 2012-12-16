#!/bin/bash


#### Add job submit details here

stencilgen -f <filename> -n <stencilsize>

metis <args> 

mpirun --hostfile hostfile.4 -np 2 ./runpde 
