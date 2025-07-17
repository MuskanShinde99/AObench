#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:15:06 2025

@author: laboptic
"""

# Import Libraries
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

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)

# Initialize Cameras

# Open the Wavefront Sensor (WFS) Camera
camera_wfs.Open()

# WFS image cropping coordinates from dao_setup
crop_size = (crop_x_start, crop_x_end, crop_y_start, crop_y_end)


#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file
print('Pupil successfully created on the SLM.')

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

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
print('New  small pupil grid created')
print('Pupil grid shape:', pupil_grid_height, pupil_grid_height)

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
#%% Create Deformable Mirror (DM)

# Initialize a deformable mirror
nact = 21  # Number of actuators
t0 = time.time()
dm_modes = make_gaussian_influence_functions(pupil_grid, (nact), pupil_size / (nact - 1), crosstalk=0.3)
deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
t1 = time.time()
print('DM created')
print("Number of DM modes =", nmodes_dm)
print(f"Time to create DM: {t1 - t0:.4f} s")

# Flatten the deformable mirror surface
deformable_mirror.flatten()
deformable_mirror.actuators[10] = 1  # Set actuator 10 to 1
plt.figure()
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface')
plt.show()
deformable_mirror.flatten()

#%% Capturing a Reference Image

# Capture a reference image using the WFS camera.
camera_wfs.Open()
time.sleep(0.3)  # Wait for stabilization
img = pylonGrab(camera_wfs, 10)
img_size = img.shape[:2]
print('WFS image size:', img_size)
plt.figure()
plt.imshow(img)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Save the reference image to a FITS file
filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(img), overwrite=True)

#%% Perform Push-Pull Calibration

# Execute the calibration procedure to acquire response images.
phase_amp = 0.1
pull_images, push_images, push_pull_images = perform_push_pull_calibration(slm, camera_wfs, img_size, deformable_mirror, phase_amp, data_pupil, pupil_mask, verbose=True)

# Save pull images to FITS files
print('Saving pull images')
filename = f'binned_response_cube_Act_only_pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(pull_images), overwrite=True)

# Save push images to FITS files
print('Saving push images')
filename = f'binned_response_cube_Act_only_push_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_images), overwrite=True)

# Save push-pull images to FITS files
print('Saving push-pull images')
filename = f'binned_response_cube_Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(push_pull_images), overwrite=True)

#%% Load Calibration Mask

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load reference image
reference_image_filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
reference_image = fits.getdata(os.path.join(folder_calib, reference_image_filename))

#%% Process Pull Images and Generate Response Matrix

# Process the pull images to obtain corrected data.
calib = 'Act_only_pull'
pull_images = np.asarray(pull_images)

# Process the pull images 
processed_pull_images = process_response_images_3s_pyr(pull_images, mask, reference_image)

# Save processed pull images as a FITS file
output_filename = f'binned_processed_response_cube_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
output_file_path = os.path.join(folder_calib, output_filename)
fits.writeto(output_file_path, processed_pull_images, overwrite=True)

# Display the first processed pull image
plt.imshow(processed_pull_images[0], cmap='gray')
plt.title('First Processed Pull Image')
plt.show()

# Create the pull response matrix
response_matrix = np.array([image[valid_pixels_indices] for image in processed_pull_images])
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

# Process the push images similarly to the pull images.
calib = 'Act_only_push'
push_images = np.asarray(push_images)

# Process the push images 
processed_push_images = process_response_images_3s_pyr(push_images, mask, reference_image)

# Save processed push images as a FITS file
output_filename = f'binned_processed_response_cube_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
output_file_path = os.path.join(folder_calib, output_filename)
fits.writeto(output_file_path, processed_push_images, overwrite=True)

# Display the first processed push image
plt.imshow(processed_push_images[0], cmap='gray')
plt.title('First Processed Push Image')
plt.show()

# Create the push response matrix
response_matrix = np.array([image[valid_pixels_indices] for image in processed_push_images])
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
calib = 'Act_push-pull'

# Convert the list of processed images into a new data cube
processed_push_pull_images = (processed_pull_images - processed_push_images)

# Save the processed push-pull images as a FITS file
output_filename = f'binned_processed_response_cube_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
output_file_path = os.path.join(folder_calib, output_filename)
fits.writeto(output_file_path, processed_push_pull_images, overwrite=True)

# Display the first processed push-pull image
plt.imshow(processed_push_pull_images[0], cmap='gray')
plt.title('First Processed Push-Pull Image')
plt.show()

# Create the response matrix 
response_matrix = np.array([image[valid_pixels_indices] for image in processed_push_pull_images])
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

# Example usage

new_cols = 3254  # Target columns after interpolation
new_rows = response_matrix.shape[0]
binned_response = bin_matrix_2d(response_matrix, new_rows, new_cols)
print(binned_response.shape)  # Expected: (441, 1627)
# Save the response matrix as a FITS file
response_matrix_filename = f'binned_response_matrix_{calib}_pup_{pupil_size}mm_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
response_matrix_file_path = os.path.join(folder_calib, response_matrix_filename)
fits.writeto(response_matrix_file_path, binned_response, overwrite=True)


