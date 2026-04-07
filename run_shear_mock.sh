#!/bin/bash

# job name
#PBS -N Cls_mock

# use this alternatively to output to the same file:
## #PBS -j oe
# std out
#PBS -o /home/hedi/PBS/
# std err
#PBS -e /home/hedi/PBS/

# No mail will be sent for the job.
##PBS -m p

# 1 node, 1 cpu, 4gb
#PBS -l select=1:ncpus=50:mem=10gb
# max time limit
#PBS -l walltime=48:00:00
#PBS -u hedi


# send email
#PBS -M di.he@ipmu.jp
#PBS -m a

# name of queue
#PBS -q small

# activate python environment
source ~/.bashrc

cd $PBS_O_WORKDIR

echo "Running field: $FIELD"
python shear_produce_mock.py "$FIELD"
