#!/bin/bash

EXEC=`hostname; ls -lah /usr/local/lib64/libfftw3.so`
STAT=$?

if [ ${STAT} -eq 0 ]
then
	echo "."
else
	echo "${EXEC}"
	return ${STAT}
fi
