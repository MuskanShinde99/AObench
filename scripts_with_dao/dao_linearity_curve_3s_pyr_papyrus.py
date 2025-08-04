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
from src.config import config
from src.dao_setup import init_setup
from src.utils import *
from src.shm_loader import shm

# Flag to identify on-sky data
OnSky = False
suffix = "_OnSky" if OnSky else ""

#Loading setup
setup = init_setup()
setup = reload_setup()

#Loading shared memories
dm_phase_shm = shm.dm_phase_shm
phase_screen_shm = shm.phase_screen_shm
phase_residuals_shm = shm.phase_residuals_shm
residual_modes_shm = shm.residual_modes_shm
slopes_image_shm = shm.slopes_image_shm
normalized_psf_shm = shm.normalized_psf_shm
computed_modes_shm = shm.computed_modes_shm
commands_shm = shm.commands_shm
strehl_ratio_shm = shm.strehl_ratio_shm
norm_flux_pyr_img_shm = shm.norm_flux_pyr_img_shm

dm_kl_modes_shm = shm.dm_kl_modes_shm
dm_act_shm = shm.dm_act_shm
dm_flat_papy_shm = shm.dm_flat_papy_shm
KL2Act_papy_shm = shm.KL2Act_papy_shm
dm_papy_shm = shm.dm_papy_shm

#Loading folder
folder_calib = config.folder_calib
folder_gui = setup.folder_gui

# Load hardware and configuration parameters
camera_wfs = setup.camera_wfs
camera_fp = setup.camera_fp
npix_small_pupil_grid = setup.npix_small_pupil_grid


#%% Creating and Displaying a Circular Pupil on the SLM

#DM set to flat
set_data_dm(setup=setup)
dm_flat_papy_shm.set_data(setup.dm_flat.astype(np.float32))
fits.writeto(folder_calib / 'dm_flat_papy.fits', setup.dm_flat.astype(np.float32), overwrite=True)


#%% Load transformation matrices

KL2Act_papy = KL2Act_papy_shm.get_data().T

#%% Load Bias Image, Calibration Mask and Interaction Matrix

# Load the bias image
bias_filename = f'bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Set bias image to zero for PAPY SIM tests
bias_image=np.zeros_like(bias_image)

# Load the calibration mask for processing images.
mask_filename = f'mask_3s_pyr{suffix}.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'response_matrix_KL2S_full_nact_{setup.nact}_amp_0.1_3s_pyr.fits'
IM_KL2S_full = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
IM_KL2S = compute_response_matrix(IM_KL2S_full, mask).astype(np.float32)

RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
print(f"Shape of the reconstruction matrix: {RM_S2KL.shape}")


#%% Load Reference Image and PSF

# Load reference image
reference_image = fits.getdata(folder_calib / 'reference_image_raw.fits')
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

#Plot
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Normalized Reference Image')
plt.show()

#%% Linearity plot KL basis: Phase KL2Act

plt.close('all')

nmodes_KL = 195
IM_KL2S = IM_KL2S[:nmodes_KL,:]
RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
print(f"Shape of the response matrix: {RM_S2KL.shape}")

# Number of KL modes to plot
num_modes = 10
applied_phase_amp = np.arange(-0.5, 0.5, 0.1) # 
computed_phase_amp = np.zeros((num_modes, len(applied_phase_amp)))

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")
    
    for i, amp in enumerate(applied_phase_amp):

        # Compute the KL mode actuator positions
        kl_mode = amp * KL2Act_papy[mode]

        # Put KL mode on the DM
        set_data_dm(kl_mode, setup=setup,)

        # Capture image and compute slopes
        slopes_image = get_slopes_image(
            mask,
            bias_image,
            normalized_reference_image,
            setup=setup,
        )

        slopes = slopes_image[valid_pixels_indices].flatten()

        # Compute modes using the response matrix
        computed_modes = slopes @ RM_S2KL 
        computed_modes_shm.set_data(np.asanyarray(computed_modes).astype(np.float32)) # setting shared memory

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
