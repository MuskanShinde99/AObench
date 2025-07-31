#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:19:52 2025

@author: laboptic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:40:46 2025

@author: laboptic
"""

# Import Libraries
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import make_gaussian_influence_functions
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

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
from src.dao_setup import init_setup
from src.utils import set_dm_actuators
setup = init_setup()  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)

# Initialize Cameras


#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file



# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))
print('Pupil successfully created on the SLM.')


#%% Create a deformable mirror (DM)

# Number of actuators
nact = 17

# Create a deformable mirror (DM)
t0 = time.time()
dm_modes = make_gaussian_influence_functions(small_pupil_grid, nact, pupil_size / (nact - 1), crosstalk=0.3)
deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
t1 = time.time()
print(f"Time to create DM: {t1 - t0:.4f} s")
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
set_dm_actuators(
    np.ones(deformable_mirror.num_actuators),
    setup=setup,
)
plt.figure()
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface')
plt.show()

#%% Create transformation matrices

# Define folder path
nmodes_zernike = 200
Act2Phs, Phs2Act = compute_Act2Phs(nact, small_pupil_grid_Npix, pupil_size, small_pupil_grid, verbose=True)
Znk2Phs, Phs2Znk = compute_Znk2Phs(nmodes_zernike, small_pupil_grid_Npix, pupil_size, small_pupil_grid, verbose=True)
Act2Znk, Znk2Act = compute_Znk2Act(nact, nmodes_zernike, Act2Phs, Phs2Act, Znk2Phs, Phs2Znk, verbose=True)

# Plot KL modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Phs[mode].reshape(small_pupil_grid_Npix, small_pupil_grid_Npix), cmap='viridis')
    axes_flat[i].set_title(f'Zernike mode {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Plot KL projections on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Act[mode].reshape(nact, nact), cmap='viridis')
    axes_flat[i].set_title(f' Zernike proj. Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

#%% Load Calibration Mask

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the bias image
bias_filename = f'binned_bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")


#%% Capturing a Reference Image

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

# Capture a reference image using the WFS camera.
camera_wfs.Open() # Open the Wavefront Sensor (WFS) Camera
time.sleep(0.3)  # Wait for stabilization of SLM
reference_image = pylonGrab(camera_wfs, 10)
normalized_reference_image = normalize_image(reference_image, mask, bias_image)

plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Normalized Reference Image')
plt.show()

# Save the reference image to a FITS file
filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(reference_image), overwrite=True)

#%% Perform Push-Pull Calibration

# Call the calibration function
phase_amp = 0.1
pull_images, push_images, push_pull_images = perform_push_pull_calibration_with_phase_basis(
    slm, camera_wfs, reference_image, mask, Znk2Act, phase_amp, data_pupil_outer, 
    data_pupil_inner, pupil_mask, small_pupil_mask, deformable_mirror, verbose=True)

# Save pull images to FITS files
print('Saving pull images')
filename = f'binned_processed_response_cube_Znk2Act_only_pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(pull_images), overwrite=True)

# Save push images to FITS files
print('Saving push images')
filename = f'binned_processed_response_cube_Znk2Act_only_push_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_images), overwrite=True)

# Save push-pull images to FITS files
print('Saving push-pull images')
filename = f'binned_processed_response_cube_Znk2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_pull_images), overwrite=True)


#%% Process Pull Images and Generate Response Matrix

calib = 'Znk2Act_only_pull'

# Create the pull response matrix
response_matrix = np.array([image[valid_pixels_indices] for image in pull_images])
response_matrix = np.array([image.ravel() for image in response_matrix])
plt.imshow(response_matrix, cmap='gray')
plt.title('Pull Response Matrix')
plt.show()

# Print the shape of the Pull Response matrix 
print("Pull Response matrix shape:", response_matrix.shape)

# Save the response matrix as a FITS file
response_matrix_filename = f'binned_response_matrix_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
response_matrix_file_path = os.path.join(folder_calib, response_matrix_filename)
fits.writeto(response_matrix_file_path, response_matrix, overwrite=True)

#%% Process Push Images and Generate Response Matrix

calib = 'Znk2Act_only_push'

# Create the push response matrix
response_matrix = np.array([image[valid_pixels_indices] for image in push_images])
response_matrix = np.array([image.ravel() for image in response_matrix])
plt.imshow(response_matrix, cmap='gray')
plt.title('Push Response Matrix')
plt.show()

# Print the shape of the Push Response matrix 
print("Push Response matrix shape:", response_matrix.shape)

# Save the response matrix as a FITS file
response_matrix_filename = f'binned_response_matrix_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
response_matrix_file_path = os.path.join(folder_calib, response_matrix_filename)
fits.writeto(response_matrix_file_path, response_matrix, overwrite=True)

#%% Generate Response Push-Pull Matrix

# Name of calib file
calib = 'Znk2Act_push-pull'

# Create the response matrix 
response_matrix = np.array([image[valid_pixels_indices] for image in push_pull_images])
response_matrix = np.array([image.ravel() for image in response_matrix])
plt.imshow(response_matrix, cmap='gray')
plt.title('Push-Pull Response Matrix')
plt.show()

# Print the shape of the Push-Pull Response matrix 
print("Push-Pull Response matrix shape:", response_matrix.shape)

# Save the response matrix as a FITS file
response_matrix_filename = f'binned_response_matrix_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
response_matrix_file_path = os.path.join(folder_calib, response_matrix_filename)
fits.writeto(response_matrix_file_path, response_matrix, overwrite=True)
