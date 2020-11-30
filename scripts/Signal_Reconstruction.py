# Main script that runs the signal reconstruction analysis
#
# Takes as input a group for comparison (e.g., 'adults')
# The subject's name for reconstruction (e.g., 'pixar002')
# The number of features used for signal reconstruction (a number between 2 and number of TRs; 10 was used in the paper)
# The train/test set order (choose 'original' or 'swapped')
# Whether this is combined for all features or each feature reconstructed separately ('yes' for all features or 'no' for reconstructing them separately)
# 
# TY 01/17/2020
#
# NOTE: ProbSRM varies by subject order. For baseline adults, subjects need to be in the same order so as to obtain the same features used for child reconstruction
# TY 09/12/2020
#############################################################

# First, import functions that we may need

import nibabel as nib
import numpy as np
import os
import sys
from brainiak import image, io
from brainiak import isfc
import brainiak.funcalign.srm
from nilearn.input_data import NiftiMasker
from scipy import stats
from scipy.stats import zscore

# From here you can import your parameters 
# including where data is stored, number of SRM iterations, and where this script should output
from SR_utils import *

#############################################################

# Set up for running the analysis

# 1. What group will be used to learn the features 
which_base=sys.argv[1]

# 2. Which subject are you reconstructing?
subj=sys.argv[2]
    
# 3. How many features will be learned?
num_feats=sys.argv[3]
num_feats=int(num_feats) #turn to integer
    
# 4. What order do you want the training and test set? Original or swapped?
trainset=sys.argv[4]
    
# 5. Are you running the main analyses (collapse across all features) or want to reconstruct each feature seperatly (note, depending on number of features, this may be time-intensive)
combine=sys.argv[5]

#############################################################

# load the brain mask (whose name is specified in SR_utils.py)
brain_mask = io.load_boolean_mask(mask_name)

# load the brain nii image
brain_nii = nib.load(mask_name)

# Make a brain masker from this mask
brain_masker=NiftiMasker(mask_img=brain_nii)
test_sub=nib.load(test_sub) # Test sub is defined in the SR_utils.py 

test_fit=brain_masker.fit(test_sub)
affine_mat = test_sub.affine
dimsize = test_sub.header.get_zooms()

# load in the functional data for the comparison group (which SRM will learn features on)
fnames = {}
images = {}
masked_images = {}
bold_base = {}
n_subjs_base = {}

fnames = get_file_names(data_dir_,which_base.lower())
    
images = io.load_images(fnames) 
masked_images = image.mask_images(images, brain_mask) 

# Concatenate all of the masked images across participants  
bold_base = image.MaskedMultiSubjectData.from_masked_images(
        masked_images, len(fnames)
    )

# Convert nans into zeros
bold_base[np.isnan(bold_base)] = 0

# reshape the data 
bold_base = np.transpose(bold_base, [1, 0, 2])

n_subjs_base = np.shape(bold_base)[-1]
print("")
print(f'Base data loaded: {which_base}\t shape: {np.shape(bold_base)}')
print("")

# load in the functional data for the individual subject that is being reconstructed
fnames_recon = {}
images_recon = {}
masked_images_recon = {}
bold_recon = {}
n_subjs_recon = {}

fnames_recon = get_individual_fn(data_dir_,subj) # Get the individual subjects name 
    
images_recon = io.load_images(fnames_recon) 
masked_images_recon = image.mask_images(images_recon, brain_mask) 

# Concatenate all of the masked images across participants  
bold_recon = image.MaskedMultiSubjectData.from_masked_images(
            masked_images_recon, len(fnames_recon)
        )

# Convert nans into zeros
bold_recon[np.isnan(bold_recon)] = 0

# reshape the data 
bold_recon = np.transpose(bold_recon, [1, 0, 2])

n_subjs_recon = np.shape(bold_recon)[-1]
print("")
print(f'Recon data loaded: {subj}\t shape: {np.shape(bold_recon)}')
print("")

#############################################################    

# Now we can do SRM!
# Fit an SRM on the baseline group of subjects and transform an individual into shared space
# What is the relationship between the held out's brain response and what the predicted brain response would using the average group response? This will serve as our measure of signal reconstruction. 
    
print('\nSTEP 0: SET UP DATA FOR SRM')

# First check if the recon subject is in the group (and needs to be removed from training)
# This is done when you are finding the baseline for a given group
same_group=0
for base_sub in range(len(fnames)):
    if subj in fnames[base_sub]:
        leftout_sub=base_sub # What is the index of the leftout subject?
        same_group=1
        
        # if we are doing reconstruction across features it's alright if the features vary slightly (which will happen anyway since this subject is removed from training)
        if combine=='yes': 
            print('Recon subject is in the baseline group; not including them in training')
            bold_base=np.delete(bold_base,base_sub,axis=2) # remove that subject from the bold for baseline 
            data_recon_base=np.dstack((bold_recon,bold_base))
            data_recon_base=np.transpose(data_recon_base,(1,0,2))
        
        # but if reconstructing individual features keep the same order so that the features are the same    
        else:
            data_recon_base=np.transpose(bold_base,(1,0,2)) 
            
        print("Reshaped data:",data_recon_base.shape)
        
# Now add this one recon subject on to the baseline group (recon sub is first in the list)
if same_group==0:        
    data_recon_base=np.dstack((bold_recon,bold_base))
    data_recon_base=np.transpose(data_recon_base,(1,0,2))
    print("Reshaped data:",data_recon_base.shape)

# now fix the labels to reflect the reshaped data
vox_num, nTR, num_subs = data_recon_base.shape  

# Preset the raw training and test data
train_data_raw = []
test_data_raw = [] 

# If you are not combining features and the subject is in the group which the features are learned on, you should put a short buffer in the test set
if combine=='no' and same_group==1: 
    
    if trainset=='swapped':
        tr_mid_low=nTR//2-10
        tr_mid_high=nTR//2
    elif trainset=='original':
        tr_mid_low=nTR//2
        tr_mid_high=nTR//2+10
    
# Otherwise, it is just the midpoint TR that separates the train and the test
else:
    tr_mid_low=nTR//2
    tr_mid_high=nTR//2

# Go through the list of subjects and separate training and test
for sub in range(num_subs): 
    
    # Do you want to swap the training and test data?
    if trainset=='swapped':
        # Take the second half of TRs as training
        train_data_raw.append(data_recon_base[:, tr_mid_high:, sub])
        # Take the first half of TRs as testing
        test_data_raw.append(data_recon_base[:, :tr_mid_low, sub])
        
    elif trainset=='original':
        # Take the first half of TRs as training
        train_data_raw.append(data_recon_base[:, :tr_mid_low, sub])
        # Take the second half of TRs as testing
        test_data_raw.append(data_recon_base[:, tr_mid_high:, sub])

# Preset the z-score version 
train_data=[]
test_data=[]

for sub in range(num_subs):    
    # Do it for training data
    train_data.append(stats.zscore(train_data_raw[sub], axis=1, ddof=1))
    # Do it for test data
    test_data.append(stats.zscore(test_data_raw[sub], axis=1, ddof=1))

# and then remember to replace nans with zero
for sub in range(num_subs):
    train_data[sub]=np.nan_to_num(train_data[sub])
    test_data[sub]=np.nan_to_num(test_data[sub])
    
# Use the number of features supplied    
features=np.int(num_feats) 

# Create the SRM object
srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)

print('\nSTEP 1: LEARN WEIGHTS AND SHARED RESPONSE FOR FIRST HALF GROUP DATA')  

# If you are reconstructing a subject and they are in the baseline group,
# AND you are reconstructing by component, you need the components to all be the same
# Therefore, you train on all the data (which is why we had a buffer earlier)
if combine=='no' and same_group==1: 
    
    # Fit all the data
    srm.fit(train_data)
    print('Learning weights of all the data to keep the components the same across individuals')
    print('Thus, we are skipping Step 2: LEARN INDIVIDUAL WEIGHT MATRIX')
    
# In all other cases, do the much cleaner version with cross validation
else:
    
    # Fit the SRM data on all of the baseline subjects (not the recon subject in position 0)
    srm.fit(train_data[1:])

    print("\nSTEP 2: LEARN INDIVIDUAL WEIGHT MATRIX\n") 

    # Now we can find the weights that would fit the held out subject into this space
    w_ = srm.transform_subject(train_data[0])

    # And we add these weights to our weight matrices
    srm.w_ = [w_] + srm.w_

print("\nSTEP 3: FIT THE SECOND HALF DATA USING LEARNED WEIGHTS")
    
# Now we can transform the test data -- for all of the subjects
shared_test=srm.transform(test_data)
print('Shape of shared test:',shared_test[0].shape)

print("\nSTEP 4: TRANSFORM GROUP AVERAGE INTO INDIVIDUAL SPACE")

# recon subject is not in the front if we are doing baseline analysis with individual features
if combine=='no' and same_group==1: 
     # remove the test subject from the list
    mask=[subjects for subjects in range(len(shared_test)) if subjects != leftout_sub]
    average_base= np.mean(np.array(shared_test)[mask,:,:],axis=0)  
    zscored_base = stats.zscore(average_base, axis=1, ddof=1) 
    
    heldout_weight = srm.w_[leftout_sub]

# otherwise we know that our recon subject is in the front
else:
    # How does the brain activity of the held out relate to that of the average baseline subject?
    average_base = np.mean(shared_test[1:],axis=0)
    zscored_base = stats.zscore(average_base, axis=1, ddof=1) 

    heldout_weight = srm.w_[0] #what was the held out's weight matrix again?
    
    
# Are you looking at all components or reconstructing individually?
if combine =='yes':
    predicted= heldout_weight.dot(zscored_base) #predicted response based on the average baseline
    actual = test_data[0] #raw held out subject data  

else: 
    # Preset the predicted and actual
    predicted=np.zeros((test_data[0].shape[0],test_data[0].shape[1],np.int(num_feats)))
    actual=np.zeros((test_data[0].shape[0],test_data[0].shape[1],np.int(num_feats)))

    for comp in range(np.int(num_feats)):
        zscored_base_comp = zscored_base[comp,:] 

        # Predict for just this component/feature
        predicted[:,:,comp]=heldout_weight[:,comp].reshape(-1,1).dot(zscored_base_comp.reshape(1,-1)) 
        actual[:,:,comp] = test_data[0] # raw held out data  (always the same)
            
# Stack them so they can be used for ISC (correlation between raw and predicted)
# If combined, this is voxels by timepoints by 2 (actual vs predicted)
# If not, this is voxels by timepoints by number of features by 2
predicted_vs_actual=np.stack((predicted,actual),axis=len(predicted.shape)) 
print('Shape of what will be correlated:',predicted_vs_actual.shape)
    
print("\nSTEP 5: CORRELATE RAW AND PREDICTED TIMECOURSE")
     
# Now run ISC! (or the correlation between actual and predicted in these ROIs)

if combine =='yes':
    intersubjectcorr=brainiak.isfc.isc(predicted_vs_actual)
    print("Resulting correlation shape:",intersubjectcorr.shape)

else:
    print('Running this separately for each component')
    # Preset the output
    comp_iscs=np.zeros((np.int(num_feats),train_data[0].shape[0]))
    
    for comp in range(np.int(num_feats)):
        comp_iscs[comp,:]=brainiak.isfc.isc(predicted_vs_actual[:,:,comp,:])
    
    print("Resulting correlation shape:",comp_iscs.shape)
        
print("\nSTEP 6: SAVE THE DATA")

# When it's over, save!

if combine =='yes':
    
    # Use the brain masker from earlier to turn this into a nifti file
    nii=brain_masker.inverse_transform(intersubjectcorr)
    
    # Add the name "baseline" if we did the baseline analysis
    if same_group==0:
        save_name=save_dir+'%s_sigrecon_%s_%s_%d.nii.gz' %(subj,which_base,trainset,num_feats)
    else:
        save_name=save_dir+'%s_sigrecon_baseline_%s_%s_%d.nii.gz' %(subj,which_base,trainset,num_feats)
    
    print(save_name)
    # Save!!!
    nib.save(nii,save_name)
    
else:
    
    # If you did not combine the data, save out all the components seperately
    for comp in range(comp_iscs.shape[0]):
        
        # Use the brain masker from earlier to turn this into a nifti file 
        nii=brain_masker.inverse_transform(comp_iscs[comp,:])
        
        # Add the name "baseline" if we did the baseline analysis
        if same_group==0:
            save_name=save_dir+'comp%d/%s_sigrecon_comp%d_%s_%s_%d.nii.gz' %(comp,subj,comp,which_base,trainset,num_feats) 
        else:
            save_name=save_dir+'comp%d/%s_sigrecon_comp%d_baseline_%s_%s_%d.nii.gz' %(comp,subj,comp,which_base,trainset,num_feats)
        
        print(save_name)
        
        # Then save!!!
        nib.save(nii,save_name)


print('FINISHED')
