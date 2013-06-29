#!/bin/sh
if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	cmake -DUSE_ICC=on -DBOOST_ROOT=/usr/local -DOPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/ -DARMADILLO_ROOT=$HOME/local -DUSE_VTK=off -DUSE_CUDA=off -DMPI_ROOT=/usr/local/openmpi/1.6/intel12/lib ..
	echo "Frodo"
elif [ $HOSTNAME = "hpc-15-35" ] ; then 
	echo "fsu mic"
	cmake -DUSE_ICC=on -DBOOST_ROOT=/usr/local -DOPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/ -DARMADILLO_ROOT=$HOME/local -DUSE_MPI=on -DUSE_VTK=off -DUSE_CUDA=off -DARMA_CBLAS=$HOME/local/usr/include/cblas.h  .. 
	echo "fsu mic"
fi
