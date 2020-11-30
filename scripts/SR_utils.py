# Utils script to be called by Signal_Reconstruction.py
# Contains custom functions that are useful for getting data and stuff
# Also contains a PARAMETERS section that can be changed according to where you stored data
#
# TY 01/14/2020

import numpy as np 
import scipy.io
import nibabel as nib
import os
import pandas as pd

######################################
######################################
##############PARAMETERS##############

# Where is the data stored?
data_dir_ = 'resampled_participants/preproc/'

# What test subject are you using for making the nifti masker (doesn't matter)
test_sub=data_dir_+'sub-pixar002_task-pixar_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'

# Where is the intersect mask stored?
mask_dir='data/'

# This mask is the intersect of all 155 subjects (122 children and 33 adults)
mask_name = os.path.join(mask_dir, 'intersect_mask_final.nii.gz')

# How many iterations for SRM?
n_iter=20

# Define where you want to save the data --> Note, you need to have made this file! 
# also, if you are looking at the feature/components separately, make sure that you make folders for each component (e.g., 'save_dir+'comp%d' % feature_number)
save_dir='analysis/subject_recons/'

######################################
######################################
######################################

# Get an individuals' file name  
def get_individual_fn(data_dir_,subj,verbose = False):
    """
    Get a single participant
    Parameters
    ----------
    subj [str]: subject's ID
    
    Return
    ----------
    fnames_ [list]: file name for that subj
    """
    fnames_=[]

    #First check if we resampled this data and saved it in the resample folder
    fname = os.path.join(
                data_dir_, 'sub-%s_task-pixar_bold_space-MNI152NLin2009cAsym_preproc.nii.gz' % (subj))
        
    # If the file exists in this directory
    if os.path.exists(fname):
            
        # add to the list of file names 
        fnames_.append(fname)
        if verbose: 
            print(fname)
        
    # If it does not exist in the directory, let them know
    else:
        print("%s not located in given data directory" % subj)
                
    return fnames_

# Get the file names 
def get_file_names(data_dir_,ages_,verbose = False):
    """
    Get all the participant
    Parameters
    ----------
    data_dir_ [str]: the data root dir
    ages_ [str]: which age group do you want to look at 
    
    Return
    ----------
    fnames_ [list]: file names for all subjs
    """
    
    fnames_ = [] # preset
    
    # What are the possible data directories? We have 2 because for some of the participants we needed to resample the data into the same voxel size as the other participants
    
    # Open the file with the participant IDs according to the ages 
    file=open('participantinfo/'+ages_+'_participants_final.txt','r')
    lines = file.readlines()
    
    # Save these 
    all_subs=[]
    for i in range(len(lines)):
        all_subs.append(lines[i].strip('\n'))
            
    for subj in all_subs: 
        
        #First check if we resampled this data and saved it in the resample folder
        fname = os.path.join(
                data_dir_, 'sub-%s_task-pixar_bold_space-MNI152NLin2009cAsym_preproc.nii.gz' % (subj))
        
        # If the file exists in this directory
        if os.path.exists(fname):
            
            # add to the list of file names 
            fnames_.append(fname)
            if verbose: 
                print(fname)
        
        # If it does not exist in the directory, let them know
        else:
            print("%s not located in given data directory" % subj)
                
    return fnames_

def get_kid_ages(ages_):
    
    # Load in the participant CSV file from open neuro
    participant_info = pd.read_csv('participantinfo/participants.csv')
    
    #There are only a few things we care about from this file, so we can condense into a smaller dataframe
    iq_info=pd.DataFrame({"ID":participant_info.participant_id, #subject ID
                          "Age":participant_info.Age, #real age
                         })               

    # Open the file with the participant IDs according to the ages 
    file=open('participantinfo/'+ages_+'_participants_final.txt','r')
    lines = file.readlines()
    
    # Save these 
    all_subs=[]
    for i in range(len(lines)):
        all_subs.append(lines[i].strip('\n'))
        
        
    file.close()
    print('Number of subjects:',len(all_subs))

    #Save out their information
    ages_kids=[]
    for i in all_subs:
        idx= np.where(iq_info.ID ==('sub-'+i))[0][0] #what's this subject's index in the panda DF?
        ages_kids.append(iq_info.Age[idx]) 
    print('Ages:',ages_kids)
    
    return ages_kids
