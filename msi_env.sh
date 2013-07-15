#source $HOME/../shared/cascade_env.sh
echo "Loading intel module" 
module load intel/2013
module load impi/intel
module load mkl/11.0.4.183
module load cmake
module load fftw
module load boost/1.53.0
module load vtk/5.4.2
module load metis

export ARMADILLO_ROOT=$HOME/../shared/intel-soft
export FFTW_ROOT=/soft/fftw/fftw-3.2.2

module list

#echo "============================"
#env
#echo "============================"

export LD_LIBRARY_PATH=/opt/intel/opencl/lib64:$LD_LIBRARY_PATH
