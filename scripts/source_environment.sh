#!/bin/bash
# 
# Set up the environment and load the modules that are necessary for running the analyses
# TY 03/22/2020

# Load the modules
module load Python/Anaconda3
module load fmriprep/1.1.8
module load brainiak/0.8-Python-Anaconda3
module load nilearn/0.5.0-Python-Anaconda3
module load OpenMPI

# Where did you load the OpenNeuro data?
DATA_DIR='data/PartlyCloudy/'

# If this does not exist, create it
if [ -e $DATA_DIR ]
then
    mkdir $DATA_DIR
fi

# Where are you putting the outputs to fMRIPrep?
FMRIPREP_DIR='data/fmriprep_outputs/'

# If this does not exist, create it
if [ -e $FMRIPREP_DIR ]
then
    mkdir $FMRIPREP_DIR
fi
