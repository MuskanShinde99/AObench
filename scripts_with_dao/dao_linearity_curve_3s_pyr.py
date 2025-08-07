#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:11:37 2025

@author: laboptic
"""

# Import Libraries
from matplotlib import pyplot as plt
import numpy as np
import time
from astropy.io import fits
import os
import dao
from src.config import config

ROOT_DIR = config.root_dir

# Import Specific Modules
import dao
from src.dao_setup import init_setup
from src.utils import set_data_dm
setup = init_setup()  # Import all variables from setup
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import *
from src.psf_centring_algorithm_functions import *
from src.scan_modes_functions import scan_othermode_amplitudes
from src.ao_loop_functions import *


#%% Creating and Displaying a Circular Pupil on the SLM

# Display Pupil Data on SLM
set_data_dm(setup=setup)

#%% Load Transformation matrices

Act2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Phs_nact_{setup.nact}_npupil_{setup.npix_small_pupil_grid}.fits'))
Phs2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Act_npupil_{setup.npix_small_pupil_grid}_nact_{setup.nact}.fits'))

# Znk2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Phs_nzernike_{nmodes_Znk}_npupil_{npix_small_pupil_grid}.fits'))
# Phs2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Znk_npupil_{npix_small_pupil_grid}_nzernike_{nmodes_Znk}.fits'))

# Znk2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Act_nzernike_{nmodes_Znk}_nact_{nact}.fits'))
# Act2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Znk_nact_{nact}_nzernike_{nmodes_Znk}.fits'))

KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{setup.nmodes_KL}_npupil_{setup.npix_small_pupil_grid}.fits'))
#Phs2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2KL_npupil_{setup.npix_small_pupil_grid}_nkl_{setup.nmodes_KL}.fits'))

KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{setup.nmodes_KL}_nact_{setup.nact}.fits'))
#Act2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2KL_nact_{setup.nact}_nkl_{setup.nmodes_KL}.fits'))


#%% Load Bias Image, Calibration Mask and Interaction Matrix

# Load the bias image
bias_filename = f'bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Load the calibration mask for processing images.
mask_filename = f'mask_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'response_matrix_KL2S_filtered_nact_{setup.nact}_amp_0.05_3s_pyr.fits'
IM_KL2S = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1

RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
print(f"Shape of the response matrix: {RM_S2KL.shape}")

#%% Load Reference Image and PSF

# Load reference image
reference_image = fits.getdata(folder_calib / 'reference_image_raw.fits')
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()
#%% Linearity plot KL basis: Phase KL2Act

nmodes_KL = 177
IM_KL2S = IM_KL2S[:nmodes_KL,:]
RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
print(f"Shape of the response matrix: {RM_S2KL.shape}")

# Number of KL modes to plot
num_modes = 10
applied_phase_amp = np.arange(-2, 2, 0.1) # 
computed_phase_amp = np.zeros((num_modes, len(applied_phase_amp)))

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")
    
    for i, amp in enumerate(applied_phase_amp):

        # Compute the KL mode actuator positions
        kl_mode = amp * KL2Act[mode]

        # Put KL mode on the DM
        set_data_dm(kl_mode, setup=setup,)

        # Capture image and compute slopes
        slopes_image = get_slopes_image(
            mask,
            bias_image,
            normalized_reference_image,
            setup=setup,
        )
        slopes_image_shm.set_data(slopes_image) # setting the shared memory separately because setting inside the function is not working

        slopes = slopes_image[valid_pixels_indices].flatten()

        # Compute modes using the response matrix
        computed_modes = slopes @ RM_S2KL 

        # Store computed modes for this mode
        computed_phase_amp[mode, i] = computed_modes[mode]

# Plot the results in subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
axes = axes.flatten()

for mode in range(num_modes):
    axes[mode].plot(applied_phase_amp, computed_phase_amp[mode], label='Computed Phase Amplitude')  # Adjust as needed
    axes[mode].plot(applied_phase_amp, applied_phase_amp, label='y = x', linestyle='--')
    axes[mode].set_xlabel('Applied Phase Amplitude [λ]')
    axes[mode].set_ylabel('Reconstructed Phase Amplitude [λ]')
    axes[mode].set_title(f'KL mode {mode}')
    axes[mode].legend()
    #axes[mode].set_ylim(-0.35, 0.35)
    axes[mode].grid(True)

plt.tight_layout()
plt.show()
