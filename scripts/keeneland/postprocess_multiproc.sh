#!/bin/bash

# Make this a function and pass these params
STENCIL_SIZES=(17 31 50 101)
TEST_CASES=(04096 06400  08100  10201  20164  27556)
NUM_PROCS=(1 2 4 6 8 10)
USE_GPU=(0 1)


function __get_summary() {
    cat ${1} | sed -n '/Verify/,//p'
}

function __get_errors() {
   echo "${1}" | sed -n '/Interior & Boundary/,$p' | sed -n "s/^.*Absolute = \([0-9][0-9a-zA-Z.*].*\),.*\( [0-9e].*\)/\1,     \2/p" 
}

function __get_line_num() {
    # $1 == input string
    # $2 == desired line
    echo "${1}" | sed -n "${2}p" 
}

function __get_benchmark() {
# $1 summary
# $2 benchmark description
# Assumes first match is what we want
echo "${1}" | grep "${2}"  |  sed -n "1p" | sed -n "s/^.*avg: .* \([0-9].*\)|.*tot: .* \([0-9].*\) |.*count= \(.*\)/\1,     \2,     \3/p"
}


for sten_size in ${STENCIL_SIZES[@]}
do 
    indx1=0
    indx2=0
    for cpu_or_gpu in ${USE_GPU[@]}
    do   
        indx1=`expr ${indx1} + 1` 


        # Start a fresh file
        echo "" > "stats_${sten_size}_use_gpu_${cpu_or_gpu}.m"

        for N in ${TEST_CASES[@]}
        do
            indx2=`expr ${indx2} + 1`
            let "indx=${indx1}*100 + ${indx2}"
            echo $indx
            
            for nprocs in ${NUM_PROCS[@]}
            do 
    

        #runlog=n${sten_size}/${N}/runlog.0
        runlog="n${sten_size}/USE_GPU_${cpu_or_gpu}/${nprocs}procs/${N}/runlog.0"
        echo "Processing ${runlog}"

        # Get summary of run: 
        summary=$( __get_summary "${runlog}" )

        # Get errors from run: 
        errors=$( __get_errors "${summary}" ) 

        # Select linf error and append it to a list
        new_l1_error=$( __get_line_num "${errors}" "1" ) 
        new_l2_error=$( __get_line_num "${errors}" "2" ) 
        new_linf_error=$( __get_line_num "${errors}" "3" ) 

        l1[${indx}]="${l1[${indx}]}\n${nprocs},     ${new_l1_error}"
        l2[${indx}]="${l2[${indx}]}\n${nprocs},     ${new_l2_error}"
        linf[${indx}]="${linf[${indx}]}\n${nprocs},     ${new_linf_error}"

        adv=$( __get_benchmark "${summary}" "Advance One" )
        rk4e=$( __get_benchmark "${summary}" "RK4 Evaluate Substep on")
        rk4adv=$( __get_benchmark "${summary}" "RK4 Advance on" )
        srecv=$( __get_benchmark "${summary}" "MPI Communicate" )

        advance[${indx}]="${advance[${indx}]}\n${nprocs},     ${adv}"
        rk4_eval[${indx}]="${rk4_eval[${indx}]}\n${nprocs},     ${rk4e}"
        rk4_adv[${indx}]="${rk4_adv[${indx}]}\n${nprocs},     ${rk4adv}"
        sendrecv[${indx}]="${sendrecv[${indx}]}\n${nprocs},      ${srecv}"

        # if the file exists then we exited prior to completion and need to resubmit
 #       if [ -z "${adv}" ]
 #       then 
 #           prev_dir=`pwd`
 #           cd "n${sten_size}/USE_GPU_${cpu_or_gpu}/${nprocs}procs"
 #           rm -r ${N}
 #           echo "Re-submitting job ${N}"
 #           qsub "submit${N}.sh"
 #           cd "${prev_dir}"
 #       fi
    done

    compiled_stats[${indx}]=$( 
        echo "\n\n\n% Absolute, Relative"  
        echo "N${N}.N=${N};"
        echo "N${N}.n=${sten_size};"
        echo "N${N}.GPU=${cpu_or_gpu};"
        echo "\n\n\n"
        echo "N${N}.l1_error=[${l1[${indx}]}\n];\n\n"
        echo "N${N}.l2_error=[${l2[${indx}]}\n];\n\n"
        echo  "N${N}.linf_error=[${linf[${indx}]}\n];\n\n"

        echo  "% Benchmarks"
        echo  "% Avg (in ms), Total, Count"
        echo  "N${N}.advance=[${advance[${indx}]}\n];\n\n"
        echo  "N${N}.rk4_eval=[${rk4_eval[${indx}]}\n];\n\n"
        echo  "N${N}.rk4_adv=[${rk4_adv[${indx}]}\n];\n\n"
        echo  "N${N}.sendrecv=[${sendrecv[${indx}]}\n];\n\n"
        echo  "\n\n\n"
        )
    echo "${compiled_stats[${indx}]}" >> "stats_${sten_size}_use_gpu_${cpu_or_gpu}.m"
    
    cell_list="${cell_list} N${N}"

        l1[${indx}]=""
        l2[${indx}]=""
        linf[${indx}]=""
        advance[${indx}]=""
        rk4_eval[${indx}]=""
        rk4_adv[${indx}]=""
        sendrecv[${indx}]=""
    done
        echo "n${sten_size}_gpu_${cpu_or_gpu} = {${cell_list}};" >> "stats_${sten_size}_use_gpu_${cpu_or_gpu}.m"
        cell_list=""
        # Index arrays by N and nproc
done
done


