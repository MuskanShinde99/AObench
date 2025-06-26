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

folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'

# #%% Accessing Devices

# # Initialize Spatial Light Modulator (SLM)
# slm = dao_setup.slm

# # Initialize Cameras
# camera_wfs = dao_setup.camera_wfs
# camera_fp = dao_setup.camera_fp


#%% Creating and Displaying a Circular Pupil on the SLM

# # Access the pupil data from the setup file
# pupil_size = dao_setup.pupil_size
# npix_small_pupil_grid = dao_setup.npix_small_pupil_grid
# small_pupil_mask = dao_setup.small_pupil_mask

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)

print('Pupil successfully created on the SLM.')

#%% Create a deformable mirror (DM)

# # Number of actuators
# nact = dao_setup.nact
# nact_valid = dao_setup.nact_valid
# nact_total = dao_setup.nact_total
# dm_modes = dao_setup.dm_modes

deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
deformable_mirror.actuators.fill(1)
plt.figure()
plt.imshow(deformable_mirror.opd.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface OPD')
plt.show()

#dao_setup.dm_act_shm.set_data(np.ones((npix_small_pupil_grid, npix_small_pupil_grid)))
#dao_setup.dm_act_shm.get_data()
 

#%% Capturing an image to check

# Display the Reference Image
reference_image = camera_wfs.get_data()
#reference_image_shm.set_data(reference_image)

plt.figure()
plt.imshow(reference_image, cmap='gray')
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Display the Focal pane image
fp_image = camera_fp.get_data()
#reference_psf_shm.set_data(fp_image)

plt.figure()
plt.imshow(fp_image) 
plt.colorbar()
plt.title('PSF')
plt.show()

#%% Creating a Flux Filtering Mask

modulation_angles = np.arange(0, 360, 10)  # angles of modulation
modulation_amp = 15 # in lamda/D

print('Starting flux modulation...')
summed_image = create_flux_filtering_mask(modulation_angles, modulation_amp, verbose=False, verbose_plot=True)
print('Flux modulation completed.')

summed_image = np.asarray(summed_image)  
   
# Display the  Summed Image
plt.figure()
plt.title('Summed Image (DM modulation)')
plt.imshow(summed_image, cmap='viridis')
plt.colorbar()
plt.savefig(os.path.join(folder_pyr_mask, f'summed_image_dm_modulation.png'))
plt.show()

#%%
# Create a filtering mask
flux_cutoff = 0.61 # Intensity cutoff threshold - 30%
mask = np.zeros(summed_image.shape, dtype=bool)
flux_limit_upper = summed_image.max() * flux_cutoff
mask[summed_image >= flux_limit_upper] = True
print('Flux filtering mask successfully created.')

# Display the Mask
plt.figure()
plt.title('Flux Filtering Mask (DM modulation)')
plt.imshow(mask, cmap='viridis')
plt.colorbar()
plt.savefig(os.path.join(folder_pyr_mask, f'mask_dm_modulation.png'))
plt.show()

# Apply Mask and Crop Results
masked_summed_image = mask*summed_image #[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3]]

# Display the Masked Summed Image
plt.figure()
plt.title('Masked Summed Image (DM modulation)')
plt.imshow(masked_summed_image, cmap='viridis')
plt.colorbar()
plt.savefig(os.path.join(folder_pyr_mask, f'masked_summed_image_dm_modulation.png'))
plt.show()

# Save the Masked Image and Mask
fits.writeto(os.path.join(folder_calib, f'binned_masked_pyr_images_pup_{pupil_size}mm_3s_pyr.fits'), np.asarray(masked_summed_image), overwrite=True)
fits.writeto(os.path.join(folder_calib, f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'), np.asarray(mask.astype(np.uint8)), overwrite=True)

# Get valid pixel from the mask
valid_pixels_indices = np.where(mask > 0)

npix_valid = valid_pixels_indices[0].shape
print(f'Number of valid pixels = {npix_valid}')

valid_pixels_mask_shm = dao.shm('/tmp/valid_pixels_mask.im.shm', np.zeros((img_size, img_size)).astype(np.uint32)) 
valid_pixels_mask_shm.set_data(mask)
#valid_pixels_indices_shm = dao.shm('/tmp/valid_pixels_mask.im.shm', np.zeros((npix_valid, 2)).astype(np.uint32))

#slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1)).astype(np.uint32)) 

KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((nmodes_KL, img_size**2)).astype(np.float32)) 
#KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((nmodes_KL, npix_valid)).astype(np.float32)) 
#S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, nmodes_KL)).astype(np.float32)) 


 #%% Computing Pupil Centers and Radii

mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
mask = mask.astype(np.uint8)  # Ensure the mask is in uint8 format

# Display the Mask
plt.figure()
plt.title('Flux Filtering Mask')
plt.imshow(mask, cmap='viridis')
plt.colorbar()
plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

# Extract pupil centers and calculate radii
pupil_centers = centroids[1:]
radii = [np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) for i in range(1, num_labels)]

print("Pupil Centers:", pupil_centers)
print("Pupil Radii:", radii)

# Pupil Centers: [[90.26202441 33.39985642]
#  [33.66571019 35.38235294]
#  [62.34477825 82.29971388]]
# Pupil Radii: [21.057199990834974, 21.064756854523722, 21.09495723828184]

#%% Centering the PSF on the Pyramid Tip

pupil_coords = pupil_centers
radius = int(round(np.mean(radii))) 

# Initial Tip-Tilt Amplitudes
tt_amplitudes = [-0.5, 0.2]  # Initial estimate for tip-tilt amplitude
focus = [0.4] #fixed focus

# Optimize Tip-Tilt Amplitudes
optimized_tt_amplitudes = optimize_amplitudes(tt_amplitudes, pupil_coords, radius)
new_ttf_amplitudes = optimized_tt_amplitudes + focus
print(f"Optimized Tip-Tilt-Focus Amplitudes: {new_ttf_amplitudes}")

# Capture Final Image After Optimization
final_img = camera_wfs.get_data() #[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3]]

# Example of plotting the cost function values after optimization
plt.figure()
plt.plot(cost_values)
plt.title("Cost Function Values Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Display Final Image
plt.figure()
plt.imshow(final_img, cmap='gray')
plt.title('Final Balanced Pupil Intensities')
plt.colorbar()
plt.show()
#%% Updating Setup File with Optimized Amplitudes

dao_setup_path = ROOT_DIR / 'src/dao_setup.py'

# Read the existing file content
with open(dao_setup_path, 'r') as file:
    content = file.read()

# Update the file with the new optimized amplitudes
new_line = f"ttf_amplitudes = {new_ttf_amplitudes}"
content = re.sub(r"ttf_amplitudes\s*=\s*\[.*?\]", new_line, content)

# Write the updated content back to the setup file
with open(dao_setup_path, 'w') as file:
    file.write(content)

print("Updated ttf_amplitudes in py")


#%%
# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)

# Display the Reference Image
time.sleep(wait_time)  # Allow the system to stabilize
reference_image = camera_wfs.get_data()
masked_reference_image = reference_image * mask
normalized_reference_image = masked_reference_image / np.abs(np.sum(masked_reference_image))
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Processed Reference Image')
plt.show()

# Display the Focal pane image
fp_image = camera_fp.get_data()
plt.figure()
plt.imshow(fp_image) 
plt.colorbar()
plt.title('PSF')
plt.show()

# Display the radial profile of the focal pane image
plt.figure()
plt.plot(fp_image[:, 253:273])
plt.title('PSF radial profile')
plt.show()


#%% Take a bias image

las.set_channel(channel)
las.enable(0) # Turn off laser
time.sleep(3)  # Allow some time for laser to turn off

# Capture and average 1000 bias frames
num_frames = 1000
bias_stack = []

for _ in range(num_frames):
    frame = camera_wfs.get_data()
    bias_stack.append(frame)

# Compute average bias image
bias_image = np.median(bias_stack, axis=0)

las.enable(1) # Turn on laser

# Save the Bias Image
fits.writeto(os.path.join(folder_calib, f'binned_bias_image.fits'), np.asarray(bias_image), overwrite=True)

# Plot
plt.figure()
plt.imshow(bias_image, cmap='gray')
plt.title('Bias image')
plt.colorbar()
plt.show()


#%% Create transformation matrices

Act2Phs, Phs2Act = compute_Act2Phs(nact, npix_small_pupil_grid, dm_modes, folder_transformation_matrices, verbose=True)

# Create KL modes
nmodes_kl = nact_valid
Act2KL, KL2Act = compute_KL2Act(nact, npix_small_pupil_grid, nmodes_kl, dm_modes, small_pupil_mask, folder_transformation_matrices, verbose=True)
KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nmodes_kl, Act2Phs, Phs2Act, KL2Act, Act2KL, folder_transformation_matrices, verbose=True)
# KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nact_valid , dm_modes, small_pupil_mask, Act2Phs, folder_transformation_matrices, verbose=True)
# Act2KL, KL2Act = compute_KL2Act(nact, nact_valid, Act2Phs, Phs2Act, KL2Phs, Phs2KL, folder_transformation_matrices, verbose=True)

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

# Plot KL modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Phs[mode].reshape(npix_small_pupil_grid, npix_small_pupil_grid)* small_pupil_mask, cmap='viridis')
    axes_flat[i].set_title(f'KL2Phs {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Create Zernike modes
nmodes_zernike = 200
Znk2Phs, Phs2Znk = compute_Znk2Phs(nmodes_zernike, npix_small_pupil_grid, pupil_size, small_pupil_grid, folder_transformation_matrices, verbose=True)
Act2Znk, Znk2Act = compute_Znk2Act(nact, nmodes_zernike, Act2Phs, Phs2Act, Znk2Phs, Phs2Znk, folder_transformation_matrices, verbose=True)

# Plot KL modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Phs[mode].reshape(npix_small_pupil_grid, npix_small_pupil_grid), cmap='viridis')
    axes_flat[i].set_title(f'Znk2Phs {mode}')
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
    axes_flat[i].set_title(f' Znk2Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

Act2Phs_shm.set_data(Act2Phs)
Phs2Act_shm.set_data(Phs2Act)

KL2Act_shm.set_data(KL2Act)
Act2KL_shm.set_data(Act2KL)
KL2Phs_shm.set_data(KL2Phs)
Phs2KL_shm.set_data(Phs2KL)

Znk2Act_shm.set_data(Znk2Act)
Act2Znk_shm.set_data(Act2Znk)
Znk2Phs_shm.set_data(Znk2Phs)
Phs2Znk_shm.set_data(Phs2Znk)

