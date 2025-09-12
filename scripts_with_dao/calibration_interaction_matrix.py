#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:29 2025

@author: laboptic
"""
# Standard library

import time
from datetime import datetime
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

#%% Load Bias Image, Calibration Mask 

# Load the bias image
bias_filename = f'bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Set bias image to zero for PAPY SIM tests
#bias_image=np.zeros_like(bias_image) #TODO: Remove it

# Load the calibration mask for processing images.
mask_filename = f'mask_3s_pyr{suffix}.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

#%% Capture Reference Image

#Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Compute and display Pupil Data on SLM
set_data_dm(setup=setup)

# Capure the Reference Image
n_frames=20
reference_image = (np.mean([camera_wfs.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_wfs.get_data().dtype)
# average over several frames
reference_image_shm.set_data(reference_image)
fits.writeto(folder_calib / 'reference_image_raw.fits', reference_image, overwrite=True)
fits.writeto(folder_calib / f'reference_image_raw_{timestamp}.fits', reference_image, overwrite=True)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Display the Focal plane image
n_frames=20
fp_image = (np.mean([camera_fp.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_fp.get_data().dtype)
reference_psf_shm.set_data(fp_image)
fits.writeto(folder_calib / 'reference_psf.fits', fp_image, overwrite=True)
fits.writeto(folder_calib / f'reference_psf_{timestamp}.fits', reference_image, overwrite=True)


#Display the PSF
plt.figure()
plt.imshow(fp_image) 
plt.colorbar()
plt.title('PSF')
plt.show()

# Display the radial profile of the PSF
plt.figure()
plt.plot(fp_image[:, 253:273])
plt.title('PSF radial profile')
plt.show()

#%%
phase_amp = 0.15

# Number of times to repeat the whole calibration
calibration_repetitions = 2

# Modes repition dictionary
#mode_repetitions = {0: 10, 3: 10} # Repeat the 0th mode ten times, the 3rd mode ten times, rest default to 1
#mode_repetitions = [2, 3]  # Repeat the 0th mode twice, the 1st mode three times, beyond the 1st default to 1

mode_repetitions = {0: 1, 1: 1}
#mode_repetitions = [200] * setup.nmodes_KL

# Run calibration and compute matrices
# use the ref img, mask directly from shared memories 
response_matrix_full, response_matrix_filtered = create_response_matrix(
    KL2Act,
    phase_amp,
    reference_image,
    mask,
    bias_image,
    verbose=True,
    verbose_plot=False,
    calibration_repetitions=calibration_repetitions,
    mode_repetitions=mode_repetitions,
    push_pull=False,
    pull_push=True,
    n_frames=1,
)

#Reset the DM to flat
set_data_dm(setup=setup)

#response_matrix_filtered = response_matrix_full[:, mask.ravel() > 0]

# Print shapes ---
print("Full response matrix shape:    ", response_matrix_full.shape)
print("Filtered response matrix shape:", response_matrix_filtered.shape)

# Plot filtered matrix 
plt.figure()
plt.imshow(response_matrix_filtered, cmap='gray', aspect='auto')
plt.title('Filtered Push-Pull Response Matrix')
plt.xlabel('Slopes')
plt.ylabel('Modes')
plt.colorbar()
plt.show()

#saving the flattened push-pull images in shared memory
KL2PWFS_cube_shm.set_data(np.asanyarray(response_matrix_full).astype(np.float64))
KL2S_shm.set_data(np.asanyarray(response_matrix_filtered).astype(np.float64))

