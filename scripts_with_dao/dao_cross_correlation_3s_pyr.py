#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:15:23 2025

@author: laboptic
"""

# Import Libraries
import gc
from tqdm import tqdm
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import make_gaussian_influence_functions, make_pupil_grid
from src.hardware import DeformableMirror
import time
from astropy.io import fits
import os
import sys
import scipy
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
from pathlib import Path
from src.config import config

ROOT_DIR = config.root_dir
folder_linearity = config.root_dir / 'outputs/Linearity_check'

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)

# Initialize Cameras

# Open the Wavefront Sensor (WFS) Camera
camera_wfs.Open()

# WFS image cropping coordinates from dao_setup
crop_size = (crop_x_start, crop_x_end, crop_y_start, crop_y_end)
crop_height = crop_size[3] - crop_size[2]  # y_end - y_start
crop_width = crop_size[1] - crop_size[0]   # x_end - x_start

#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file
print('Pupil successfully created on the SLM.')

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

#%% Create a deformable mirror (DM)

# Number of actuators
nact = 21

# Create a deformable mirror (DM)
t0 = time.time()
dm_modes = make_gaussian_influence_functions(pupil_grid, nact, pupil_size / (nact - 1), crosstalk=0.3)
deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
t1 = time.time()
print(f"Time to create DM: {t1 - t0:.4f} s")
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
deformable_mirror.actuators.fill(1)
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface')
plt.show()

#%% Create a pupil grid for a smaller pupil area

# Generate a grid for computing phase basis 
dataWidth = 1920
dataHeight = 1200
pixel_size = 8e-3  # pixel size in mm 
Npix_pupil = int(pupil_size / pixel_size)  # Convert pupil size to pixels

# Set up pupil grid dimensions
pupil_grid_width = int(Npix_pupil * 1.1)
pupil_grid_height = int(Npix_pupil * 1.1)
ngrid = pupil_grid_width
pupil_grid = make_pupil_grid([pupil_grid_width, pupil_grid_height], [pupil_grid_width * pixel_size, pupil_grid_height * pixel_size])

# Calculate offsets to center the pupil grid with respect to the SLM grid
offset_height = (dataHeight - pupil_grid_height) // 2
offset_width = (dataWidth - pupil_grid_width) // 2

# Create a grid mask for visualization
grid_mask = np.zeros((dataHeight, dataWidth), dtype=bool)
grid_mask[offset_height:offset_height + pupil_grid_height, offset_width:offset_width + pupil_grid_width] = 1

plt.figure()
plt.imshow(grid_mask)
plt.colorbar()
plt.title('Grid Mask')
plt.show()

#%% Load Transformation matrices

nmodes_zernike = 400
nmodes_kl = 350
nact=21

# Define folder path
Act2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Phs_nact_{nact}_npupil_{ngrid}.fits'))
Phs2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Act_npupil_{ngrid}_nact_{nact}.fits'))

Znk2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Phs_nzernike_{nmodes_zernike}_npupil_{ngrid}.fits'))
Phs2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2Znk_npupil_{ngrid}_nzernike_{nmodes_zernike}.fits'))

Znk2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'Znk2Act_nzernike_{nmodes_zernike}_nact_{nact}.fits'))
Act2Znk = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2Znk_nact_{nact}_nzernike_{nmodes_zernike}.fits'))

KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{nmodes_kl}_npupil_{ngrid}.fits'))
Phs2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2KL_npupil_{ngrid}_nkl_{nmodes_kl}.fits'))

KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'))
Act2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'))

#%% Load Calibration Mask and Response Matrix

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'response_matrix_Znk2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_Znk2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
RM_PyWFS2Znk = np.linalg.pinv(IM_Znk2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2Znk.shape}")

# Load the response matrix 
IM_filename = f'binned_response_matrix_KL2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_KL2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
IM_KL2PyWFS = IM_KL2PyWFS[:nmodes_kl,:]
RM_PyWFS2KL = np.linalg.pinv(IM_KL2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2KL.shape}")

#%% Capute the dark image 

# # Capture a dark image using the WFS camera.
# camera_wfs.Open()
# time.sleep(0.3)  # Wait for stabilization
# dark_img = pylonGrab(camera_wfs, 10)

# plt.figure()
# plt.imshow(dark_img)
# plt.colorbar()
# plt.title('Dark Image')
# plt.show()

# # Save the reference image to a FITS file
# filename = f'binned_dark_img_pup_{pupil_size}mm_3s_pyr.fits'
# fits.writeto(os.path.join(folder, filename), np.asarray(dark_img), overwrite=True)

#%% Capute the refrence image 

from matplotlib.colors import LogNorm

# Capture a reference image using the WFS camera.
camera_wfs.Open()
time.sleep(0.3)  # Wait for stabilization
img = pylonGrab(camera_wfs, 10)
reference_image = img #- dark_img
masked_reference_image = reference_image * mask
normalized_reference_image = masked_reference_image / np.sum(masked_reference_image)

plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Save the reference image to a FITS file
# filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
# fits.writeto(os.path.join(folder, filename), np.asarray(reference_image), overwrite=True)

#%%

# Number of Zernike modes to plot
num_modes = 350
phase_amp = 0.1  # Fixed phase amplitude
cross_corr = np.zeros((num_modes, num_modes))  # Store all computed modes

# Initialize an array to store the cube of normalized pyr images
pyr_images_cube = []
pyr_images_cube_raw =[]

# Loop through each Zernike mode
for mode in range(num_modes):
    
    print(f"Applying phase for mode: {mode}")

    # Put the Zernike mode on the DM
    deformable_mirror.flatten()
    #deformable_mirror.actuators = phase_amp * KL2Act[mode]

    # Create and update SLM data with current phase settings
    data_dm = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_dm[:, :] = deformable_mirror.surface.shaped

    # Combine data_zernike and data
    data_slm = data_pupil + data_dm
    data_slm[pupil_mask] = ((data_slm[pupil_mask] * 256) % 256)

    # Show data on SLM:
    slm.set_data(data_slm.astype(np.uint8))

    # Capture image
    time.sleep(0.3)  # Wait for 0.3 seconds
    pyr_img = pylonGrab(camera_wfs, 1)
    
    #Substract dark image
    #pyr_img = pyr_img - dark_img

    # Apply the mask
    masked_pyr_img = pyr_img * mask

    # Normalize by the sum of the masked image
    normalized_pyr_img = masked_pyr_img / np.sum(masked_pyr_img)

    # Subtract the reference image
    normalized_pyr_img = normalized_pyr_img - normalized_reference_image

    # Extract valid pixels from the cropped mask
    new_pyr_img = normalized_pyr_img[valid_pixels_indices]
    new_pyr_img = new_pyr_img.flatten()

    # Compute modes using the response matrix
    computed_modes = new_pyr_img @ RM_PyWFS2KL

    # Store all computed modes
    cross_corr[mode, :] = computed_modes
    
    # Append the normalized pyr image to the cube
    pyr_images_cube.append(normalized_pyr_img)

# Convert the list to a numpy array for the cube of images
pyr_images_cube = np.array(pyr_images_cube)
pyr_images_cube_raw = np.array(pyr_img)

# Save the cube to a FITS file
filename_cube = f'raw_background_pyr_images_cube.fits'
fits.writeto(os.path.join(folder_linearity, filename_cube), pyr_images_cube_raw, overwrite=True)

filename_cube = f'background_pyr_images_cube_ref_img_substracted_dark_substracted.fits'
fits.writeto(os.path.join(folder_linearity, filename_cube), pyr_images_cube, overwrite=True)

#%%
   
# Save to a FITS file
filename = 'KL_basis_background_ref_img_substracted_dark_substracted.fits'#f'KL_basis_cross-corr_binned_4x4_nact_21_nmodes_{num_modes}.fits'
fits.writeto(os.path.join(folder_linearity, filename), np.asarray(cross_corr/4), overwrite=True)

# Plot the computed phase amplitude matrix
plt.figure(figsize=(8, 6))
plt.imshow(cross_corr/4, cmap='viridis', aspect='auto')
plt.colorbar(label='Computed Phase Amplitude')
plt.xlabel('Computed Mode')
plt.ylabel('Applied Mode')
plt.title('KL basis Background')
plt.show()