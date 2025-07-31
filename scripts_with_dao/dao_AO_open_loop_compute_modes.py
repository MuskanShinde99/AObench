#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:29 2025

@author: laboptic
"""
# Standard library
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from matplotlib.colors import LogNorm

# Import Specific Modules
from src.utils import *
from src.shm_loader import shm

setup = init_setup()
dm_phase_shm = shm.dm_phase_shm
phase_screen_shm = shm.phase_screen_shm
phase_residuals_shm = shm.phase_residuals_shm
residual_modes_shm = shm.residual_modes_shm
slopes_image_shm = shm.slopes_image_shm
normalized_psf_shm = shm.normalized_psf_shm
computed_modes_shm = shm.computed_modes_shm
commands_shm = shm.commands_shm
dm_kl_modes_shm = shm.dm_kl_modes_shm
dm_act_shm = shm.dm_act_shm
KL2Act_papy_shm = shm.KL2Act_papy_shm

#%% Creating and Displaying a Circular Pupil on the SLM

# Display Pupil Data on SLM
set_data_dm(setup=setup)

#%% Load transformation matrices

# From folder 
# KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{setup.nmodes_KL}_nact_{setup.nact}.fits'))
# KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{setup.nmodes_KL}_npupil_{setup.npix_small_pupil_grid}.fits'))

# # From shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

KL2Act_papy = KL2Act_papy_shm.get_data()

#%% Load Bias Image, Calibration Mask and Interaction Matrix

# Load the bias image
bias_filename = f'binned_bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'binned_response_matrix_KL2S_filtered_nact_{setup.nact}_amp_0.1_3s_pyr.fits'
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

# Load Diffraction limited PSF
diffraction_limited_psf = fits.getdata(folder_calib / 'reference_psf.fits')
#diffraction_limited_psf /= diffraction_limited_psf.sum()
fp_img_shape = diffraction_limited_psf.shape
print('PSF shape:', fp_img_shape)

#Radial Profile 
plt.figure()
plt.plot(diffraction_limited_psf[:,291:311:2])
plt.plot(diffraction_limited_psf[286:316:2,:].T)
plt.show()

# Create the PSF mask 
psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)

plt.figure()
plt.imshow(psf_mask)
plt.colorbar()
plt.title('PSF Mask to compute strehl')
plt.show()

# Integrate the flux in that small region
integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()
print('sum center PSF =', integrated_diff_psf)

# Plot PSF with selected region overlayed
plt.figure()
plt.imshow(diffraction_limited_psf, norm=LogNorm(), cmap='viridis')
plt.colorbar(label='Intensity')
plt.title('PSF with Selected Region')
circle = plt.Circle((psf_center[1], psf_center[0]), radius=50, color='red', fill=False, linewidth=2) # Overlay the integration region (circle)
plt.gca().add_patch(circle)
plt.show()

# Compute the Strehl ratio
#strehl_ratio = integrated_obs / integrated_diff

#%% Defining the number of Kl modes used for Closed lop simulations

plt.close('all')

# Define KL modes to consider
nmodes_kl = 195
KL2Act_papy_new = KL2Act_papy[:nmodes_kl, :]
Act2KL_papy_new = scipy.linalg.pinv(KL2Act_papy_new)
IM_KL2S_new = IM_KL2S[:nmodes_kl, :]
RM_S2KL_new = np.linalg.pinv(IM_KL2S_new, rcond=0.10)


#%%
# Load hardware and configuration parameters
camera_wfs = setup.camera_wfs
camera_fp = setup.camera_fp
npix_small_pupil_grid = setup.npix_small_pupil_grid

# Load the folder
folder_gui = setup.folder_gui

# Flatten the DM
set_data_dm(setup=setup)
#deformable_mirror.flatten()

# Reference image 
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape

print('Reference image shape:', pyr_img_shape)

# Diffraction limited PSF
diffraction_limited_psf = diffraction_limited_psf.astype(np.float32)
diffraction_limited_psf /= diffraction_limited_psf.sum()
fp_img_shape = diffraction_limited_psf.shape

print('PSF shape:', fp_img_shape)

# Create the PSF mask 
psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)
# Integrate the flux in that small region
integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()

print('sum center PSF =', integrated_diff_psf)

# Get valid pixel indices from the mask
valid_pixels_indices = np.where(mask > 0)

 #%%   
# Initialize arrays to store Strehl ratio and total residual phase
# strehl_ratios = np.zeros(num_iterations)
# residual_phase_stds = np.zeros(num_iterations)


i = 0
while True:
    # Capture and process WFS image
    slopes_image = get_slopes_image(
        mask,
        bias_image,
        normalized_reference_image,
        setup=setup,
    )
    slopes_image_shm.set_data(slopes_image)
    slopes = slopes_image[valid_pixels_indices].flatten()
    #fits.writeto(os.path.join(folder_gui, f'slopes_image.fits'), slopes_image, overwrite=True)
    
    # Compute KL modes present
    computed_modes = slopes @ RM_S2KL_new
    # multiply by two because this mode is computed for DM surface and we want DM phase
    computed_modes_shm.set_data(computed_modes) # setting shared memory
    
    # Compute actuator commands
    act_pos = computed_modes @ KL2Act_papy_new
    commands_shm.set_data(act_pos) # setting shared memory

    # Capture PSF
    fp_img = camera_fp.get_data()
    fp_img = np.maximum(fp_img, 1e-10)
    normalized_psf_shm.set_data((fp_img / np.max(fp_img))) # setting shared memory
    #fits.writeto(os.path.join(folder_gui, f'normalized_psf.fits'), (fp_img / np.max(fp_img)), overwrite=True)

    # Compute Strehl ratio
    observed_psf = fp_img / fp_img.sum()
    integrated_obs_psf = observed_psf[psf_mask].sum()
    strehl_ratio = integrated_obs_psf / integrated_diff_psf
    # strehl_ratio = np.max(observed_psf) / np.max(diffraction_limited_psf)
    # strehl_ratios[i] = strehl_ratio

    i += 1  # increment loop index


