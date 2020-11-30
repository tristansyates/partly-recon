## Scripts folder

To run our main analyses using the default settings, first make sure that you have updated the parameters in SR_utils.py. Then, simply submit supervisor_sigrecon as a job. This script iterates through all of the child subjects, submitting a separate run_sigrecon.sh job for each, which runs the python script Signal_Reconstruction.py. This script saves out a nifti file with the voxelwise signal reconstruction values for that subject.

You can change the default variables as follows:

    base_group and recon_group
    options: 'adults', 'all_children', 'fours', 'fives, 'sixseven', 'eightnine', 'tenup'

    feats
    options: an integer in quotes ranging from 2 to 81 (cannot be larger than half of the TRs in the movie)

    trainset
    options: 'original' (first half of data used for training, second half for testing) or 'swapped' (second half of data used for training, first half for testing)

    combine
    options: 'yes' (combine across all features in the shared space) or 'no' (reconstruct each shared feature separately)

Note that you can also make different files in the participant_info folder if you are interested in using a different participant group than the ones we have already made (e.g., perhaps you want to reconstruct features from females to males, or vice versa). Simply make a new file called participant_info/$NAME_participants_final.txt that contains all of the subject IDs. You can then call that new $NAME as an option for either the base_group or recon_group variable. 
