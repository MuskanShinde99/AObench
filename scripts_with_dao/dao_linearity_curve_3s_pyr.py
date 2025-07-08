#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:11:37 2025

@author: laboptic
"""

# Import Libraries
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
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
from matplotlib.colors import LogNorm
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil


#%% Creating and Displaying a Circular Pupil on the SLM

# Display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)

#%% Load Transformation matrices

Act2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Phs_nact_{nact}_npupil_{small_pupil_grid_Npix}.fits'))
Phs2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Act_npupil_{small_pupil_grid_Npix}_nact_{nact}.fits'))

Znk2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Phs_nzernike_{nmodes_zernike}_npupil_{small_pupil_grid_Npix}.fits'))
Phs2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Znk_npupil_{small_pupil_grid_Npix}_nzernike_{nmodes_zernike}.fits'))

Znk2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Act_nzernike_{nmodes_zernike}_nact_{nact}.fits'))
Act2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Znk_nact_{nact}_nzernike_{nmodes_zernike}.fits'))

# Extend Zernike basis for the SLM
Znk2Phs_extended = np.zeros((nmodes_zernike, dataHeight, dataWidth), dtype=np.float32)
Znk2Phs_extended[:, offset_height:offset_height + small_pupil_grid_Npix, offset_width:offset_width + small_pupil_grid_Npix] = Znk2Phs.reshape(nmodes_zernike, small_pupil_grid_Npix, small_pupil_grid_Npix)

KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{nmodes_kl}_npupil_{small_pupil_grid_Npix}.fits'))
Phs2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2KL_npupil_{small_pupil_grid_Npix}_nkl_{nmodes_kl}.fits'))

KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'))
Act2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'))

# Extend KL basis for the SLM
KL2Phs_extended = np.zeros((nmodes_kl, dataHeight, dataWidth), dtype=np.float32)
KL2Phs_extended[:, offset_height:offset_height + small_pupil_grid_Npix, offset_width:offset_width + small_pupil_grid_Npix] = KL2Phs.reshape(nmodes_kl, small_pupil_grid_Npix, small_pupil_grid_Npix)


#%% Load Calibration Mask and Response Matrix

# Load the bias image
bias_filename = f'binned_bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix
IM_filename = f'binned_response_matrix_Znk2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_Znk2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
RM_PyWFS2Znk = np.linalg.pinv(IM_Znk2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2Znk.shape}")

# Load the response matrix 
IM_filename = f'binned_response_matrix_KL2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_KL2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
IM_KL2PyWFS = IM_KL2PyWFS[:nmodes_kl,:]
RM_PyWFS2KL = np.linalg.pinv(IM_KL2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2KL.shape}")

#%% Capute the refrence image 

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

# Capture a reference image using the WFS camera.
camera_wfs.Open() # Open the Wavefront Sensor (WFS) Camera
time.sleep(0.3)  # Wait for stabilization of SLM
reference_image = pylonGrab(camera_wfs, 10)
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Save the reference image to a FITS file
filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(reference_image), overwrite=True)

#%% Linearity plot KL basis: Phase KL2Act

phase_amp = 0.1
nmodes_kl = 177
IM_KL2PyWFS = IM_KL2PyWFS[:nmodes_kl,:]
RM_PyWFS2KL = np.linalg.pinv(IM_KL2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2KL.shape}")

# Number of Zernike modes to plot
num_modes = 1
applied_phase_amp = np.arange(-1, 1.1, 0.1)
computed_phase_amp = np.zeros((num_modes, len(applied_phase_amp)))

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")
    
    for i, amp in enumerate(applied_phase_amp):

        # Put the Zernike mode on the DM
        deformable_mirror.flatten()
        deformable_mirror.actuators = amp * KL2Act[mode]

        # Create and update SLM data with current phase settings
        data_dm = np.zeros((small_pupil_grid_Npix, small_pupil_grid_Npix), dtype=np.float32)
        data_dm[:, :] = deformable_mirror.opd.shaped

        # Combine data_zernike and data
        data_slm = data_pupil_outer.copy()
        data_inner = ((data_pupil_inner + data_dm) * 256) % 256
        data_slm[pupil_mask] = data_inner[small_pupil_mask]

        # Show data on SLM:
        slm.set_data(data_slm.astype(np.uint8))

        # Capture image
        time.sleep(0.11)  # Wait for 0.3 seconds
        pyr_img = pylonGrab(camera_wfs, 1)

        # Process the Pyramid image
        normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
        slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)
        slopes = slopes_image[valid_pixels_indices].flatten()

        # Compute modes using the response matrix
        computed_modes = slopes @ RM_PyWFS2KL

        # Store computed modes for this mode
        computed_phase_amp[mode, i] = computed_modes[mode]

# Plot the results in subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
axes = axes.flatten()

for mode in range(num_modes):
    axes[mode].plot(applied_phase_amp, computed_phase_amp[mode], label='Computed Phase Amplitude')  # Adjust as needed
    axes[mode].plot(applied_phase_amp, applied_phase_amp, label='y = x', linestyle='--')
    axes[mode].set_xlabel('Applied Phase Amplitude [2π rad ptp]')
    axes[mode].set_ylabel('Reconstructed Phase Amplitude [2π rad ptp]')
    axes[mode].set_title(f'KL mode {mode}')
    axes[mode].legend()
    
    axes[mode].grid(True)

plt.tight_layout()
plt.show()

#%% Linearity plot Zenike basis: Phase Znk2Act

# Number of Zernike modes to plot
num_modes = 2
applied_phase_amp = np.arange(-3, 3, 0.1)
computed_phase_amp = np.zeros((num_modes, len(applied_phase_amp)))

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")
    
    for i, amp in enumerate(applied_phase_amp):

        # Put the Zernike mode on the DM
        deformable_mirror.flatten()
        deformable_mirror.actuators = amp * Znk2Act[mode]

        # Create and update SLM data with current phase settings
        data_dm = np.zeros((small_pupil_grid_Npix, small_pupil_grid_Npix), dtype=np.float32)
        data_dm[:, :] = deformable_mirror.surface.shaped

        # Combine data_zernike and data
        data_slm = data_pupil_outer.copy()
        data_inner = ((data_pupil_inner + data_dm) * 256) % 256
        data_slm[pupil_mask] = data_inner[small_pupil_mask]

        # Show data on SLM:
        slm.set_data(data_slm.astype(np.uint8))

        # Capture image
        time.sleep(0.3)  # Wait for 0.3 seconds
        pyr_img = pylonGrab(camera_wfs, 1)

        # Process the Pyramid image
        normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
        slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)
        slopes = slopes_image[valid_pixels_indices].flatten()

        # Compute modes using the response matrix
        computed_modes = slopes @ RM_PyWFS2Znk

        # Store computed modes for this mode
        computed_phase_amp[mode, i] = computed_modes[mode]

# Plot the results in subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
axes = axes.flatten()

for mode in range(num_modes):
    axes[mode].plot(applied_phase_amp, computed_phase_amp[mode], label='Computed Phase Amplitude')  # Adjust as needed
    axes[mode].plot(applied_phase_amp, applied_phase_amp, label='y = x', linestyle='--')
    axes[mode].set_xlabel('Applied Phase Amplitude [2π rad ptp]')
    axes[mode].set_ylabel('Reconstructed Phase Amplitude [2π rad ptp]')
    axes[mode].set_title(f'Zernike mode {mode}')
    axes[mode].legend()
    
    axes[mode].grid(True)

plt.tight_layout()
plt.show()


#%% Linearity plot Zenike basis: Phase Znk2Phs

# Select the mode
num_modes = 5

applied_phase_amp = np.arange(-1, 1., 0.2)
computed_phase_amp = []

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")
    
    for i, amp in enumerate(applied_phase_amp):

        # Put the zernike mode 
        deformable_mirror.flatten()
        data_zernike = np.zeros((dataHeight, dataWidth), dtype=np.float32)
        data_zernike[:, :] = amp*Znk2Phs_extended[mode].reshape(dataHeight, dataWidth)
    
        # Combine data_zernike and data
        data_slm = (data_pupil + data_zernike) 
        data_slm[pupil_mask] = ((data_slm[pupil_mask] * 256) % 256)

        # Show data on SLM:
        slm.set_data(data_slm.astype(np.uint8))

        # Capture image
        time.sleep(0.3)  # Wait for 0.3 seconds
        pyr_img = pylonGrab(camera_wfs, 1)

        # Process the Pyramid image
        normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
        slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)
        slopes = slopes_image[valid_pixels_indices].flatten()

        # Compute modes using the response matrix
        computed_modes = slopes @ RM_PyWFS2Znk

        # Store computed modes for this mode
        computed_phase_amp[mode, i] = computed_modes[mode]

# Plot the results in subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
axes = axes.flatten()

for mode in range(num_modes):
    axes[mode].plot(applied_phase_amp, computed_phase_amp[mode], label='Computed Phase Amplitude')  # Adjust as needed
    axes[mode].plot(applied_phase_amp, applied_phase_amp, label='y = x', linestyle='--')
    axes[mode].set_xlabel('Applied Phase Amplitude')
    axes[mode].set_ylabel('Reconstructed Phase Amplitude')
    axes[mode].set_title(f'Zernike mode {mode}')
    axes[mode].legend()
    
    axes[mode].grid(True)

plt.tight_layout()
plt.show()