#!/bin/bash

prev_dir=${PWD}

for d in 50
    #17 31 
    #50 101
do

    for e in 04096  05184  06400  07225  08100  09216  10201  15129  20164  25600  27556
    #for e in 01024  02116  03136  04096  05184  06400  07225  08100  09216  10201  15129  20164  25600  27556
#    for e in 01024  02116  03136  04096  05184  06400  
#    for e in 07225  08100  09216  10201  
    #for e in 15129  20164  
#    for e in 25600  27556
    #for e in 4096
    do
        # this allows us to reuse the weights (potentially)
        # 0 = CPU, 1 = Block GPU, 2 = Thread GPU (per stencil)
        for gpu_type in 0 1 2
        do
            rundir="n${d}/USE_GPU_${gpu_type}"
            mkdir -p ${rundir}
            runconf="test_${d}_md${e}.conf"
            echo "${rundir}/${runconf}"

            f=`ls ~/GRIDS/md/*${e}`

            echo "GRID_FILENAME=${f}" > ${rundir}/${runconf} 
        #    echo "STENCIL_SIZE=$d" >> ${rundir}/${runconf}
            echo "USE_GPU=${gpu_type}" >> ${rundir}/${runconf}
            cat confs/test${d}.conf >> ${rundir}/${runconf}

cat > "${rundir}/submit${e}.sh" <<EOF 
#!/bin/sh
#PBS -N n${d}_md${e}_vortex_roll
#PBS -j oe
#PBS -q batch
#PBS -A UT-NTNL0051

### (USED) Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##

#PBS -l walltime=00:45:00
## THIS IS FOUR PROCS TOTAL (i.e., 2x2): 
#PBS -l nodes=1:ppn=1
##PBS -l feature=cudaworkshop 

### End of PBS options ###

date
cd \$PBS_O_WORKDIR

echo "nodefile="
cat \$PBS_NODEFILE
echo "=end nodefile"

export CL_KERNELS=${prev_dir}/cl_kernels

# run the program

which mpirun
# mca option binds processes to lowest numbered processor and locks it in until
# termination (i.e. no swtiching)
mpirun --mca mpi_paffinity_alone 1 ${prev_dir}/vortex_rollup_on_sphere.x -c ${runconf} -o runlog -d "${prev_dir}/${rundir}/${e}"

date

# no need to keep bulk of files. We just want benchmarks and final accuracy
rm ${e}/*.ascii ${e}/*.vtk ${e}/*.mtx ${e}/FINAL_SOLUTION.txt 

EOF

            cd ${rundir}
            qsub submit${e}.sh

            cd ${prev_dir}
        done
    done
done


