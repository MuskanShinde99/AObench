#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:20 2025

@author: laboptic
"""

# Standard library
import os
import time
from datetime import datetime
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import importlib

# Import Specific Modules
import dao
from src.dao_setup import init_setup, las
from src.utils import set_data_dm, reload_setup

#Loading setup
setup = init_setup()
setup = reload_setup()

from src.config import config
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import * 
from src.psf_centring_algorithm_functions import *
from src.shm_loader import shm
from src.scan_modes_functions import *
from src.ao_loop_functions import *

#Loading folder
folder_calib = config.folder_calib

#Loading shared memories
bias_image_shm = shm.bias_image_shm
valid_pixels_mask_shm = shm.valid_pixels_mask_shm
npix_valid_shm = shm.npix_valid_shm
reference_image_shm = shm.reference_image_shm
normalized_ref_image_shm = shm.normalized_ref_image_shm
reference_psf_shm = shm.reference_psf_shm
KL2Act_papy_shm = shm.KL2Act_papy_shm
dm_flat_papy_shm = shm.dm_flat_papy_shm
dm_papy_shm = shm.dm_papy_shm


# #%% Accessing Devices

# # Initialize Cameras
# camera_wfs = setup.camera_wfs
# camera_fp = setup.camera_fp

# # Intialize DM
# deformable_mirror = setup.defomable_mirror
    
#%%
# Load the bias image
bias_filename = f'bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

#%% Setting DM to flat

# Compute and display Pupil Data on SLM
# data_slm = compute_data_slm()
# slm.set_data(data_slm)
# time.sleep(wait_time)

# DM set to flat
set_data_dm(setup=setup)
#dm_flat_papy_shm.set_data(setup.dm_flat.astype(np.float32))
#fits.writeto(folder_calib / 'dm_flat_papy.fits', setup.dm_flat.astype(np.float32), overwrite=True)

#%% Load transformation matrices

# # Load transformation matrices from shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

# From folder 
# KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{setup.nmodes_KL}_nact_{setup.nact}.fits'))
# KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{setup.nmodes_KL}_npupil_{setup.npix_small_pupil_grid}.fits'))

KL2Act_papy = KL2Act_papy_shm.get_data().T

# plt.figure()
# plt.plot(KL2Act_papy[1,:])
# plt.show()

#%% Creating a Flux Filtering Mask

method='dm_random'
flux_cutoff = 0.08 #0.00000001 # 0.08 - papy dm random; 0.2 - geneva dm random
modulation_angles = np.arange(0, 2000, 1)  # angles of modulation
modulation_amp = 2 # in lamda/D
n_iter=500 # number of iternations for dm random commands

mask = create_flux_filtering_mask(method, flux_cutoff, KL2Act_papy[0], KL2Act_papy[1],
                               modulation_angles, modulation_amp, n_iter, 
                               create_summed_image=False, verbose=False, verbose_plot=True,
                               OnSky=False, )

print(f"Mask dimensions: {mask.shape}")

valid_pixels_mask_shm.set_data(mask)

# Get valid pixels from the mask
#put with in the function to create mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape[0]
npix_valid_shm.set_data(np.array([[npix_valid]]))
print(f'Number of valid pixels = {npix_valid}')

#Reset the DM to flat
set_data_dm(setup=setup)

# Create shared memories that depends on number of valid pixels
KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((setup.nmodes_KL, setup.img_size_wfs_cam_x*setup.img_size_wfs_cam_y), dtype=np.float64))
slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1), dtype=np.uint64))
KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((setup.nmodes_KL, npix_valid), dtype=np.float64))
S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, setup.nmodes_KL), dtype=np.float64))


#%% Centering the PSF on the Pyramid Tip

# print('Start centering algorithm')
# center_psf_on_pyramid_tip(mask=mask, 
#                           bounds = [(-2.0, 2.0), (-2.0, 2.0)], variance_threshold=0.1, 
#                           update_setup_file=True, verbose=True, verbose_plot=True)

 #%% Scanning modes to find zero of the pyramid

# test_values = np.arange(-0.5, 0.5, 0.05)
# mode_index = 3 # 0 - focus, 1 - astimgatism, 2 -astigmatism 
# #scan_othermode_amplitudes(test_values, mode_index, update_setup_file=True)
# scan_othermode_amplitudes_wfs_std(test_values, mode_index, mask, 
#                                   update_setup_file=False)
  
# #revise the crieteria to standard deviation of intensities within the valid pixels

#%% Capture Reference Image

#Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Compute and display Pupil Data on SLM
set_data_dm(setup=setup)

# Capure the Reference Image
n_frames=1000
reference_image = (np.mean([camera_wfs.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_wfs.get_data().dtype)
#reference_image = fits.getdata(os.path.join(folder_calib, f'mask_3s_pyr.fits'))
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
phase_amp = 0.05

# Number of times to repeat the whole calibration
calibration_repetitions = 1

# Modes repition dictionary
#mode_repetitions = {0: 10, 3: 10} # Repeat the 0th mode ten times, the 3rd mode ten times, rest default to 1
#mode_repetitions = [2, 3]  # Repeat the 0th mode twice, the 1st mode three times, beyond the 1st default to 1

mode_repetitions = [50]*150
#mode_repetitions = [200] * setup.nmodes_KL

# Run calibration and compute matrices
# use the ref img, mask directly from shared memories 
response_matrix_full, response_matrix_filtered = create_response_matrix(
    KL2Act_papy,
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
    n_frames=100,
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

