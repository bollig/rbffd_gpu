#source $HOME/../shared/cascade_env.sh
echo "Loading intel module" 
module load intel/2013
module load impi/intel
module load mkl/11.0.4.183
module load cmake
module load fftw
module load boost
module load vtk/5.4.2
module load metis

export ARMADILLO_ROOT=$HOME/../shared/intel-soft

module list

#echo "============================"
#env
#echo "============================"

export LD_LIBRARY_PATH=/opt/intel/opencl/lib64:$LD_LIBRARY_PATH