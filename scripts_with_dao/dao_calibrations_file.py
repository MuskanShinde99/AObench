#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:20 2025

@author: laboptic
"""

# Import Libraries
from matplotlib.colors import LogNorm
import gc
from tqdm import tqdm
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import *
import time
from astropy.io import fits
import os
import sys
import scipy
import matplotlib.animation as animation
from pathlib import Path
from datetime import datetime

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
import dao
from src.dao_setup import *  # Import all variables from setup
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import * 
from src.psf_centring_algorithm_functions import *
from src.create_shared_memories import *
from src.scan_modes_functions import *
from src.ao_loop_functions import *

# #%% Accessing Devices

# # Initialize Spatial Light Modulator (SLM)
# slm = dao_setup.slm

# # Initialize Cameras
# camera_wfs = dao_setup.camera_wfs
# camera_fp = dao_setup.camera_fp

# # Intialize DM
# deformable_mirror = dao_setup.defomable_mirror

#%% Take a bias image

# Turn off laser
las.enable(0) 
time.sleep(2)  # Allow some time for laser to turn off

# Capture and average 1000 bias frames
n_frames=1000
bias_image = np.median([camera_wfs.get_data() for i in range(n_frames)], axis=0)
bias_image_shm.set_data(bias_image)

# Plot
plt.figure()
plt.imshow(bias_image, cmap='gray')
plt.title('Bias image')
plt.colorbar()
plt.show()

# Save the Bias Image
fits.writeto(os.path.join(folder_calib, f'binned_bias_image.fits'), np.asarray(bias_image), overwrite=True)

#%% Turn on laser

las.enable(1) 
time.sleep(5)

#%% Creating and Displaying a Circular Pupil on the SLM

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)

print('Pupil created on the SLM.')

#DM function for papyrus, and have a dm flat
# for a dm flat put 10% of the total DM unit
# make setting the shared memory part of the DM function

#%% Capturing an image to check

# Display the Reference Image
reference_image = camera_wfs.get_data()
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Display the Focal pane image
fp_image = camera_fp.get_data()
plt.figure()
plt.imshow(np.log10(fp_image))
plt.colorbar()
plt.title('PSF')
plt.show()


# Find the max position
max_pos = np.unravel_index(np.argmax(fp_image), fp_image.shape)
print(f"Maximum value is at position: {max_pos}, value: {fp_image[max_pos]}")

# Extract vertical slice through the PSF peak column
x_col = max_pos[1]
profile = fp_image[:, x_col]
y = np.arange(fp_image.shape[0])

# Compute half-maximum value
max_val = profile[max_pos[0]]
half_max = max_val / 2

# Plot the profile with horizontal lines
plt.figure()
plt.plot(y, profile, label='PSF profile')
plt.axhline(max_val, color='r', linestyle='--', label='Max')
plt.axhline(half_max, color='g', linestyle=':', label='Half Max')
plt.title('PSF radial profile (vertical slice)')
plt.xlabel('Y pixel')
plt.ylabel('Intensity')
plt.legend()
plt.show()


#%% Load transformation matrices

# # Load transformation matrices from shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

# From folder 
KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_KL}_nact_{nact}.fits'))
KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{nmodes_KL}_npupil_{npix_small_pupil_grid}.fits'))


plt.figure()
plt.imshow(KL2Act[3,:].reshape(nact,nact))
plt.colorbar()
plt.title('KL mode')
plt.show()

# plt.figure()
# plt.imshow(small_pupil_mask*KL2Phs[0,:].reshape(npix_small_pupil_grid,npix_small_pupil_grid))
# plt.colorbar()
# plt.title('KL mode')
# plt.show()



#%% Creating a Flux Filtering Mask

method='tip_tilt_modulation'
flux_cutoff = 0.4
modulation_angles = np.arange(0, 360, 10)  # angles of modulation
modulation_amp = 15 # in lamda/D
n_iter=200 # number of iternations for dm random commands

mask = create_flux_filtering_mask(method, flux_cutoff, 
                               modulation_angles, modulation_amp, n_iter,
                               create_summed_image=False, verbose=False, verbose_plot=True)

valid_pixels_mask_shm.set_data(mask)

# Get valid pixels from the mask
#put with in the function to create mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape[0]
npix_valid_shm.set_data(np.array([[npix_valid]]))
print(f'Number of valid pixels = {npix_valid}')

#Reset the DM to flat
slm.set_data(data_slm)

# Create shared memories that depends on number of valid pixels
KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((nmodes_KL, img_size_wfs_cam**2)).astype(np.float32)) 
slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1)).astype(np.uint32)) 
KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((nmodes_KL, npix_valid)).astype(np.float32)) 
S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, nmodes_KL)).astype(np.float32)) 

#%% Centering the PSF on the Pyramid Tip

plt.close('all')

center_psf_on_pyramid_tip(mask=mask, initial_tt_amplitudes=[-0.2, 0.1], 
                          bounds = [(-2.0, 2.0), (-2.0, 2.0)], variance_threshold=0.01, 
                          update_setup_file=True, verbose=True, verbose_plot=True)

#%% Scanning modes to find zero of the pyramid

test_values = np.arange(-0.5, 0.5, 0.05)
mode_index = 1 # 0 - focus, 1 - astimgatism, 2 -astigmatism 
#scan_othermode_amplitudes(test_values, mode_index, update_setup_file=True)
scan_othermode_amplitudes_wfs_std(test_values, mode_index, mask, update_setup_file=True)
  
#revise the crieteria to standard deviation of intensities within the valid pixels

#%% Capture Reference Image

# put functions to capture and average the frames from the camera

#Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)  # Allow the system to stabilize

# Capure the Reference Image
n_frames=20
reference_image = (np.mean([camera_wfs.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_wfs.get_data().dtype)
# average over several frames
reference_image_shm.set_data(reference_image)
fits.writeto(folder_calib / 'reference_image_raw.fits', reference_image, overwrite=True)
fits.writeto(folder_calib / f'reference_image_raw_{timestamp}.fits', reference_image, overwrite=True)

# Normailzed refrence image
normalized_reference_image = normalize_image(reference_image, mask, bias_img=np.zeros_like(reference_image))
normalized_ref_image_shm.set_data(normalized_reference_image)
fits.writeto(folder_calib / 'reference_image_normalized.fits', normalized_reference_image, overwrite=True)
fits.writeto(folder_calib / f'reference_image_normalized_{timestamp}.fits', normalized_reference_image, overwrite=True)

#Plot
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Normalized Reference Image')
plt.show()

# Display the Focal plane image
n_frames=20
fp_image = (np.mean([camera_fp.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_fp.get_data().dtype)
reference_psf_shm.set_data(fp_image)
fits.writeto(folder_calib / 'reference_psf.fits', fp_image, overwrite=True)

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
phase_amp = 1

# Number of times to repeat the whole calibration
calibration_repetitions = 1

# Modes repition dictionary
#mode_repetitions = {0: 10, 3: 10} # Repeat the 0th mode ten times, the 3rd mode ten times, rest default to 1
#mode_repetitions = [2, 3]  # Repeat the 0th mode twice, the 1st mode three times, beyond the 1st default to 1

mode_repetitions = {0: 10, 1: 10, 10: 5}

# Run calibration and compute matrices
# use the ref img, mask directly from shared memories 
response_matrix_full, response_matrix_filtered = create_response_matrix(
    KL2Act,
    phase_amp,
    reference_image,
    mask,
    verbose=True,
    verbose_plot=False,
    calibration_repetitions=calibration_repetitions,
    mode_repetitions=mode_repetitions  
)

#Reset the DM to flat
slm.set_data(data_slm)

#response_matrix_filtered = response_matrix_full[:, mask.ravel() > 0]

#saving the flattened push-pull images in shared memory
KL2PWFS_cube_shm.set_data(response_matrix_full)
KL2S_shm.set_data(response_matrix_filtered)

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

