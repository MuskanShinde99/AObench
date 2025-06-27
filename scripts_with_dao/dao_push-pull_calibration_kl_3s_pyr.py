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
from hcipy import *
import time
from astropy.io import fits
import os
import sys
import scipy
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *

folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'

#%% Creating and Displaying a Circular Pupil on the SLM

# Display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)
print('Pupil created on the SLM.')

#%% Create a deformable mirror (DM)

deformable_mirror = DeformableMirror(dm_modes_full)
nmodes_dm = deformable_mirror.num_actuators
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()

#%% Load transformation matrices

nmodes_kl = nact_valid
KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'))
Act2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'))

# Plot KL projections on actuators
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

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)  # Wait for stabilization of SLM

# Capture a reference image using the WFS camera.
reference_image = camera_wfs.get_data()
normalized_reference_image = normalize_image(reference_image, mask, bias_image)

plt.figure()
plt.imshow(reference_image)
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
    KL2Act, phase_amp, reference_image, mask, verbose=True)

# Save pull images to FITS files
print('Saving pull images')
filename = f'binned_processed_response_cube_KL2Act_only_pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(pull_images), overwrite=True)

# Save push images to FITS files
print('Saving push images')
filename = f'binned_processed_response_cube_KL2Act_only_push_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_images), overwrite=True)

# Save push-pull images to FITS files
print('Saving push-pull images')
filename = f'binned_processed_response_cube_KL2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_pull_images), overwrite=True)


#%% Process Pull Images and Generate Response Matrix

calib = 'KL2Act_only_pull'

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

calib = 'KL2Act_only_push'

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
calib = 'KL2Act_push-pull'

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
