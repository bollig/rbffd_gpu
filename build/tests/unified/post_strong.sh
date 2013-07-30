#!/bin/bash -l 
dir="strong_*_${1}_*"
for d in `/bin/ls -d ${dir}`; do grep  "SpMV + " ${d}/time_log.derivs.* | awk '{sum+=$7} END { print NR, "," , sum/NR}'; done | sort -n
	
	for d in `/bin/ls -d ${dir}`; do grep  "perform SpMV" ${d}/time_log.spmvtest.* | awk '{sum+=$6} END { print NR, ",", sum/NR}'; done | sort -n

for d in `/bin/ls -d ${dir}`; do grep  "Synchronize" ${d}/time_log.spmvtest.* | awk '{sum+=$6} END { print NR, "," , sum/NR}'; done | sort -n
