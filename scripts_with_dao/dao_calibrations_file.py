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

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
import dao
from src.dao_setup import *  # Import all variables from setup
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.calibration_functions import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *
from src.create_shared_memories import *

folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'
folder_closed_loop_tests = ROOT_DIR / 'outputs/Closed_loop_tests'
folder_turbulence = ROOT_DIR / 'outputs/Phase_screens'

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
num_frames = 1000
bias_stack = []

for _ in range(num_frames):
    frame = camera_wfs.get_data()
    bias_stack.append(frame)

# Compute average bias image
bias_image = np.median(bias_stack, axis=0)
bias_image_shm.set_data(bias_image)
# Turn on laser
las.enable(1) 
time.sleep(5)

# Plot
plt.figure()
plt.imshow(bias_image, cmap='gray')
plt.title('Bias image')
plt.colorbar()
plt.show()

# Save the Bias Image
fits.writeto(os.path.join(folder_calib, f'binned_bias_image.fits'), np.asarray(bias_image), overwrite=True)

#%% Creating and Displaying a Circular Pupil on the SLM

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)

print('Pupil created on the SLM.')

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

#reset the DM to flat

#%% Create shared memories that depends on number of valid pixels

# Get valid pixels from the mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape[0]
print(f'Number of valid pixels = {npix_valid}')

KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((nmodes_KL, img_size_wfs_cam**2)).astype(np.float32)) 
slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1)).astype(np.uint32)) 
KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((nmodes_KL, npix_valid)).astype(np.float32)) 
S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, nmodes_KL)).astype(np.float32)) 

#%% Centering the PSF on the Pyramid Tip


center_psf_on_pyramid_tip(mask=mask, initial_tt_amplitudes=[0, 0], focus=[0.4], bounds = [(-2.0, 2.0), (-2.0, 2.0)],
                              update_setup_file=True, verbose=True, verbose_plot=True)

# average more images
# recover the last value I stopped the optimization at
# maybe display the tip tilt values instead of the pupil instensities

#%% Scanning modes to find zero of the pyramid


#%% Create transformation matrices

Act2Phs, Phs2Act = compute_Act2Phs(nact, npix_small_pupil_grid, dm_modes_full, folder_transformation_matrices, verbose=True)

# Create KL modes
nmodes_kl = nact_valid
Act2KL, KL2Act = compute_KL2Act(nact, npix_small_pupil_grid, nmodes_kl, dm_modes, small_pupil_mask, folder_transformation_matrices, verbose=True)
KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nmodes_kl, Act2Phs, Phs2Act, KL2Act, Act2KL, folder_transformation_matrices, verbose=True)

# Plot KL projected| on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Act[mode].reshape(nact, nact), cmap='viridis')
    axes_flat[i].set_title(f' KL2Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Act2Phs_shm.set_data(Act2Phs)
# Phs2Act_shm.set_data(Phs2Act)

KL2Act_shm.set_data(KL2Act)
# Act2KL_shm.set_data(Act2KL)
KL2Phs_shm.set_data(KL2Phs)
# Phs2KL_shm.set_data(Phs2KL)

#%% Capture Reference Image

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)  # Allow the system to stabilize

# Capure the Reference Image
reference_image = camera_wfs.get_data()
# average over several frames
reference_image_shm.set_data(reference_image)
fits.writeto(folder_calib / 'reference_image_raw.fits', reference_image, overwrite=True)

# Normailzed refrence image
normalized_reference_image = normalize_image(reference_image, mask, bias_img=np.zeros_like(reference_image))
fits.writeto(folder_calib / 'reference_image_normalized.fits', normalized_reference_image, overwrite=True)

#Plot
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Normalized Reference Image')
plt.show()

# Display the Focal plane image
fp_image = camera_fp.get_data()
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

#%% Perform Push-Pull Calibration

# Call the calibration function
phase_amp = 0.1
pull_images, push_images, push_pull_images = perform_push_pull_calibration_with_phase_basis(
    KL2Act, phase_amp, reference_image, mask, verbose=True)

# Save pull images to FITS files
print('Saving pull images')
filename = f'binned_processed_response_cube_KL2PWFS_only_pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(pull_images), overwrite=True)

# Save push images to FITS files
print('Saving push images')
filename = f'binned_processed_response_cube_KL2PWFS_only_push_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_images), overwrite=True)

# Save push-pull images to FITS files
print('Saving push-pull images')
filename = f'binned_processed_response_cube_KL2PWFS_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_pull_images), overwrite=True)


#%% Compute interaction matrix

from datetime import datetime
# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create full and filtered matrices ---
response_matrix_full = compute_response_matrix(push_pull_images)
response_matrix_filtered = compute_response_matrix(push_pull_images, mask=mask)

KL2PWFS_cube_shm.set_data(response_matrix_full)

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

# Save matrices (standard and timestamped)

# Filtered
filtered_base = f'binned_response_matrix_KL2S_filtered_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
filtered_timestamped = f'binned_response_matrix_KL2S_filtered_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr_{timestamp}.fits'
fits.writeto(os.path.join(folder_calib, filtered_base), response_matrix_filtered, overwrite=True)
fits.writeto(os.path.join(folder_calib, filtered_timestamped), response_matrix_filtered, overwrite=True)
print(f"Filtered matrix saved to:\n  {filtered_base}\n  {filtered_timestamped}")

# Full
full_base = f'binned_response_matrix_KL2S_full_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
full_timestamped = f'binned_response_matrix_KL2S_full_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr_{timestamp}.fits'
fits.writeto(os.path.join(folder_calib, full_base), response_matrix_full, overwrite=True)
fits.writeto(os.path.join(folder_calib, full_timestamped), response_matrix_full, overwrite=True)
print(f"Full matrix saved to:\n  {full_base}\n  {full_timestamped}")