#!/bin/bash

STEN=50

N=10
echo ${N}
mkdir -p ${N}_cubed 
cd ${N}_cubed
../gen_reg_grid.x -x ${N} -y ${N} -z ${N} -s ${STEN}  -l 100
cd -

for N in $(seq 20 20 200)
do
	echo ${N}
	mkdir -p ${N}_cubed 
	cd ${N}_cubed
	../gen_reg_grid.x -x ${N} -y ${N} -z ${N} -s ${STEN} -l 100
	cd -
done
