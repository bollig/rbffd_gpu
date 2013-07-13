#!/bin/bash -l 
dir="strong_*_${1}_*"
for d in `/bin/ls -d ${dir}`; do grep  "SpMV + Is" ${d}/time_log.derivs.* | awk '{sum+=$7} END { print NR, "," , sum/NR}'; done | sort -n
	
	for d in `/bin/ls -d ${dir}`; do grep  "Compute Der" ${d}/time_log.derivs.* | awk '{sum+=$6} END { print NR, ",", sum/NR}'; done | sort -n

for d in `/bin/ls -d ${dir}`; do grep  "Synchronize" ${d}/time_log.derivs.* | awk '{sum+=$8} END { print NR, "," , sum/NR}'; done | sort -n
