#!/usr/bin/env bash

#SBATCH --output=fmriprep-%j.out
#SBATCH -p verylong
#SBATCH -t 3-12
#SBATCH --mem 100G

# Source the environment
./source_environment.sh

SUB=$1

# First we need to make sure that any instance of IsRunning is deleted
rm ${FMRIPREP_DIR}/sub-${SUB}/scripts/IsRunning*

# Run fmriprep on the indicated subject
fmriprep ${DATA_DIR} ${FMRIPREP_DIR} participant --participant_label $SUB --ignore fieldmaps 
