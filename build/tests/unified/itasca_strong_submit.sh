#!/bin/bash -l 

for NODES in 1 2 4 8 16 32 64 128
do 

for STEN_SIZE in 17 31 50 101
do
	EXEC_FILE=itasca_strong_${STEN_SIZE}_${NODES}.pbs

cat > ${EXEC_FILE}  << EOF 
#!/bin/bash -l
#PBS -l walltime=4:00:00,nodes=${NODES}:ppn=8,pmem=2500mb
#PBS -q batch
#PBS -m ae 
#PBS -N strong_${STEN_SIZE}_impi
## PBS -e error.\$PBS_JOBID
## PBS -o output.\$PBS_JOBID


echo ------------------------------------------------------
echo -n 'Job is running on node '; cat \$PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on \$PBS_O_HOST
echo PBS: originating queue is \$PBS_O_QUEUE
echo PBS: executing queue is \$PBS_QUEUE
echo PBS: working directory is \$PBS_O_WORKDIR
echo PBS: execution mode is \$PBS_ENVIRONMENT
echo PBS: job identifier is \$PBS_JOBID
echo PBS: job name is \$PBS_JOBNAME
echo PBS: node file is \$PBS_NODEFILE
echo PBS: current home directory is \$PBS_O_HOME
echo PBS: PATH = \$PBS_O_PATH
echo ------------------------------------------------------

#env |grep PBS

#source \$HOME/../shared/cascade_env.sh
echo "Loading intel module" 
module load intel/2013
module load impi/intel
module load mkl/11.0.4.183
module load cmake
module load fftw
module load boost/1.53.0
module load vtk/5.4.2
module load metis

export ARMADILLO_ROOT=\$HOME/../shared/intel-soft

module list

#echo "============================"
#env
#echo "============================"

export LD_LIBRARY_PATH=/opt/intel/opencl/lib64:\$LD_LIBRARY_PATH

date

TEST_TYPE=strong
MD=165
NPERDIM=160
N=4096000
STEN_SIZE=50
GRID_FILE=\$HOME/GRIDS/regular/\${NPERDIM}_cubed/regulargrid_\${NPERDIM}x_\${NPERDIM}y_\${NPERDIM}z_final.ascii

PROC_LIST=\$PBS_NP

if [ "\$PBS_NP" = "8" ]; then
	PROC_LIST="1 2 4 8"
fi
#500 1000

for NPROC in \$PROC_LIST
do
	MY_MPI_EXE="mpirun -r ssh -l -np \${NPROC}"


	METIS_FILE=metis_stencils.graph.part.\${NPROC}
	JOB_RAN_FILE=job_ran

	NEW_WORKDIR=\$PBS_O_WORKDIR/\${TEST_TYPE}_\${N}_${STEN_SIZE}_\${NPROC}proc

	# If NPROC is 1 then we cant use MPIRUN
	if [ "\$NPROC" = "1" ]; then
		MY_MPI_EXE=""
	fi

	mkdir -p \$NEW_WORKDIR
	cd \$NEW_WORKDIR

	if [ ! -f \$JOB_RAN_FILE ] 
	then

		# Generate a regular grid of size 100^3 
		#./gen_regular_grid.x -x 100 -y 100 -z 100

		# Read grid, generate stencils (Note: -c 4 is required because MD nodes have 4 cols)
		\$PBS_O_WORKDIR/sten_gen.x -g \${GRID_FILE} -c 3 -N \${N} -n ${STEN_SIZE} -w 0 -l 100

		echo "sten_gen Exit status: \$?"
		#sleep 5s

		# Decompose domain for parallel run into NPROC partitions
		gpmetis metis_stencils.graph \${NPROC}

		echo "gpmetis Exit status: \$?"
		#sleep 5s

		\${MY_MPI_EXE} hostname
		echo "hostname Exit status: \$?"

		# add -l for verbose logging of output 
		\${MY_MPI_EXE} \$PBS_O_WORKDIR/compute_weights.x -w 15 -g input_grid.ascii -N \${N} -n ${STEN_SIZE} --eps_c1 0.035 --eps_c2 0.1 -p metis_stencils.graph.part.\${NPROC}

		echo "compute_weights Exit status: \$?"
		#sleep 5s
		touch \$JOB_RAN_FILE
	fi

	\${MY_MPI_EXE} \$PBS_O_WORKDIR/evaluate_derivatives.x -g input_grid.ascii -N \${N}  -n ${STEN_SIZE} --eps_c1=0.035 --eps_c2=0.1 -w 15 

	echo "evaluate_derivatives Exit status: \$?"

	rm *.ascii *.bmtx *.mtx 
	rm metis_stencils*

	echo "Done with cleanup" 

	#echo "demonstrating 2D regular grid: " 

	#./sten_gen.x -g regulargrid_10x_10y_1z_final.ascii -N 100 -n 20 -c 3
	#./compute_weights.x -g input_grid.ascii -N 100 -n 20 --eps_c1 0 --eps_c2 0.3 -a -D 2
	#./evaluate_derivatives.x -g input_grid.ascii -N 10 -n 20 --eps_c1=0.035 --eps_c2=0.1 -w 15 -a -D 2


	#mpirun -output-filename runlog -np 2 ./compute_weights.x -g input_grid.ascii -c 4 -N 4096 -n 17 --eps_c1 0.035 --eps_c2 0.1 -p lsh_100_stencils.graph.part.2

	date
	cd - 
done

EOF

#	qsub ${EXEC_FILE}
done 
done
