#!/bin/bash

# Original: 
# qsub -pe make_gts_rr $1 -N p${1}_heat ./run.sh

########################################     
# New commands provided by Mickey: 
#  
#  
#  We are going to a new queue structure on acm, which will make it easier to
#  target specific types of GPU cards.  Previously, we had different distinct
#  queues for the different types of cards.  Now, we will have just one queue
#  named "all.q".  It will offer two parallel environments (PEs):  fill_up and
#  round_robin.  There will be complexes defined for each type of GPU available in
#  the cluster.  The list of defined complexes is currently: tesla1060, tesla870,
#  gtx275, and 8500gt.  There are 16 nodes containing Tesla 1060's, 3 nodes with
#  Tesla 870's, and one node each for GTX 275 and 8500 GT.  Most GPU users will
#  want to use one or both of the Tesla complexes.
#
#
#      (1) Queue all.q with 4 slots/node
#      (2) PEs fill_up and round_robin
#      (2) Compexes tesla870, tesla1060, gtx275, 8500gt
#
#
#Typical submissions might include:
#
# tie up 1 slot
#qsub  -l tesla1060   run.sh
#
#
# round robin on _available_ slots
#qsub -pe round_robin 8 run.sh
#
#
# fills up on _available_ slots
#qsub -pe fill_up 8  run.sh
#

# round robin tying up $1 total slots on machines with tesla1060
# NOTE: $1 is the argv[1] input to this script
qsub -pe round_robin $1 -l tesla1060 -N p${1}_heat ./run.sh

#
# fill up 4 total total slots of tesla1060
#qsub -pe fill_up 4 -l tesla1060 run.sh
#
#
#lock all slots on a node
#qsub -pe threaded 4 -l tesla1060 run.sh
