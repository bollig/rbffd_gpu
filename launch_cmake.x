#!/bin/sh
if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	cmake -DUSE_MPI=off -DUSE_ICC=on -DBOOST_ROOT=/usr/local -DOPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/ -DARMADILLO_ROOT=$HOME/local -DUSE_VTK=off -DUSE_CUDA=off ..
	echo "Frodo"
#----------------------
elif [ $HOSTNAME = "hpc-15-35" ] ; then 
	echo "fsu mic"
	cmake -DUSE_ICC=on -DBOOST_ROOT=/usr/local -DOPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/ -DARMADILLO_ROOT=$HOME/local -DUSE_MPI=on -DUSE_VTK=off -DUSE_CUDA=off -DARMA_CBLAS=$HOME/local/usr/include/cblas.h  .. 
	echo "fsu mic"
#---------------------
elif [ $HOSTNAME = "casiornis" ] ; then 
	echo "casiornis"
	cmake -DUSE_ICC=on -DBOOST_ROOT=$BOOST_ROOT -DOPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/ -DARMADILLO_ROOT=$HOME/local -DUSE_MPI=on -DUSE_VTK=off -DUSE_CUDA=off -DARMA_CBLAS=$HOME/local/usr/include/cblas.h  .. 
	echo "casiornis"
#----------------------
fi

