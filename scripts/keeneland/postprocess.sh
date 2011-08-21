#!/bin/bash

# Make this a function and pass these params
STENCIL_SIZES=(17)
TEST_CASES=(04096 05184)
USE_GPU=(0 1)


function __get_summary() {
    cat ${1} | sed -r -n -e '/Verify/,${p}'
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
#sed -n '/Advance One/,$p' |  sed -n "1p" | 
#sed -n "s/^.*avg: .* \([0-9].*\)|.*tot: .* \([0-9].*\) |.*count= \(.*\)/\1,     \2, \3/p"
}


for cpu_or_gpu in ${USE_GPU[@]}
    do

        l1=""
        l2=""
        linf=""
        advance=""
        rk4_eval=""
        rk4_adv=""
        sendrecv=""

        for sten_size in ${STENCIL_SIZES[@]}
    do 
        for N in ${TEST_CASES[@]}
    do

            #runlog=n${sten_size}/${N}/runlog.0
            runlog="n${sten_size}/USE_GPU_${cpu_or_gpu}/${N}/runlog.0"
            echo "Processing ${runlog}"

            # Get summary of run: 
            summary=$( __get_summary "${runlog}" )

            # Get errors from run: 
            errors=$( __get_errors "${summary}" ) 

            # Select linf error and append it to a list
            new_l1_error=$( __get_line_num "${errors}" "1" ) 
            new_l2_error=$( __get_line_num "${errors}" "2" ) 
            new_linf_error=$( __get_line_num "${errors}" "3" ) 

            l1="${l1}\n${new_l1_error}"
            l2="${l2}\n${new_l2_error}"
            linf="${linf}\n${new_linf_error}"

            adv=$( __get_benchmark "${summary}" "Advance One" )
            rk4e=$( __get_benchmark "${summary}" "Advance One" )
            rk4adv=$( __get_benchmark "${summary}" "Advance One" )
            srecv=$( __get_benchmark "${summary}" "Advance One" )

            advance="${advance}\n${adv}"
            rk4_eval="${rk4_eval}\n${rk4e}"
            rk4_adv="${rk4_adv}\n${rk4adv}"
            sendrecv="${sendrecv}\n${srecv}"

        done

     compiled_stats=$( 
        echo -e "% Absolute, Relative"  
        echo -e "n${sten_size}.l1_error=[${l1}\n];\n\n"
        echo -e "n${sten_size}.l2_error=[${l2}\n];\n\n"
        echo -e "n${sten_size}.linf_error=[${linf}\n];\n\n"

        echo -e "% Benchmarks"
        echo -e "% Avg (in ms), Total, Count"
        echo -e "n${sten_size}.advance=[${advance}\n];\n\n"
        echo -e "n${sten_size}.rk4_eval=[${rk4_eval}\n];\n\n"
        echo -e "n${sten_size}.rk4_adv=[${rk4_adv}\n];\n\n"
        echo -e "n${sten_size}.sendrecv=[${sendrecv}\n];\n\n"
    )

    echo "${compiled_stats}" > "stats_${sten_size}_use_gpu_${cpu_or_gpu}.m"

    done
done

   
