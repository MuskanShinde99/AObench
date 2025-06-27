#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:20 2025

@author: laboptic
"""

# Import Libraries
import os
import time
import re
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from astropy.io import fits
from hcipy import *
import dao
import sys
from pathlib import Path

# Configure root paths 
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from DEVICES_3.Basler_Pylon.test_pylon import *

#import src.dao_setup as dao_setup  # Import the setup file
from src.dao_setup import *
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.create_transformation_matrices import *
#from src.create_shared_memories import *

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

# Turn on laser
las.enable(1) 
time.sleep(2)

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
flux_cutoff = 0.35
modulation_angles = np.arange(0, 360, 10)  # angles of modulation
modulation_amp = 15 # in lamda/D
n_iter=200 # number of iternations for dm random commands

mask = create_flux_filtering_mask(method, flux_cutoff, 
                               modulation_angles, modulation_amp, n_iter,
                               create_summed_image=True, verbose=False, verbose_plot=True)

valid_pixels_mask_shm = dao.shm('/tmp/valid_pixels_mask.im.shm', np.zeros((img_size, img_size)).astype(np.uint32)) 
valid_pixels_mask_shm.set_data(mask)

#reset the DM to flat

#%% Create shared memories that depends on number of valid pixels

# Get valid pixels from the mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape
print(f'Number of valid pixels = {npix_valid}')

# slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1)).astype(np.uint32)) 
# KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((nmodes_KL, img_size**2)).astype(np.float32)) 
# KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((nmodes_KL, npix_valid)).astype(np.float32)) 
# S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, nmodes_KL)).astype(np.float32)) 

#%% Centering the PSF on the Pyramid Tip

center_psf_on_pyramid_tip(mask=mask, initial_tt_amplitudes=[0, 0], focus=[0.4], 
                              update_setup_file=True, verbose=True, verbose_plot=True)

#average more images
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

# KL2Act_shm.set_data(KL2Act)
# Act2KL_shm.set_data(Act2KL)
# KL2Phs_shm.set_data(KL2Phs)
# Phs2KL_shm.set_data(Phs2KL)

#%% Capture Reference Image

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)  # Allow the system to stabilize

# Capure the Reference Image
reference_image = camera_wfs.get_data()
# average over several frames
#reference_image_shm.set_data(reference_image)
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
#reference_psf_shm.set_data(fp_image)
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
