#!/usr/bin/env bash
# Input python command to be submitted as a job
#
# This is the run script for finding the signal reconstruction for individual subjects
# Can be run on it's own to reconstruct an individual subject, or is called by the supervisor script
# To reconstruct a full set of subjects for a given group
#
#SBATCH --output=SigRecon-%j.out
#SBATCH -p short
#SBATCH -t 1:59:00
#SBATCH --mem 100G

# Source the environment
./source_environment.sh

# Update the relevant variables, which should be given as inputs to this script or be delivered via the supervisor script
base_group=$1 # Which group will be the baseline? (e.g., 'adults')
subj=$2 # What is the subject name?? (e.g., 'pixar001')
feats=$3 # How many features for running SRM?  (must be an integer in quotes, e.g., '10')
trainset=$4 # Which training order are you using ('original' or 'swapped')
combine=$5 # Are you combining features or reconstructing each feature individually? (yes or no)

# Run the script 

python scripts/Signal_Reconstruction.py $base_group $subj $feats $trainset $combine
