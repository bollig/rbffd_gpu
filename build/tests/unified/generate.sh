#!/bin/bash

STEN=16

N=10
echo ${N}
mkdir -p ${N}_cubed 
cd ${N}_cubed
../gen_reg_grid.x -x ${N} -y ${N} -z ${N} -s ${STEN} 
cd ../
#$(seq 20 20 200)
for N in 10
do
	echo ${N}
	mkdir -p ${N}_cubed 
	cd ${N}_cubed
	../../gen_reg_grid.x -x ${N} -y ${N} -z ${N} -s ${STEN} -l 10
	cd ../
done
