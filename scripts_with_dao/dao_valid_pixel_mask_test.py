#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:59:41 2025

@author: ristretto-dao
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

# Set the working directory
os.chdir('/home/ristretto-dao/optlab-master')
from DEVICES_3.Basler_Pylon.test_pylon import *

# Set the Working Directory
os.chdir('/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO')

# Import Specific Modules
import src.dao_setup as dao_setup  # Import the setup file
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.create_transformation_matrices import *

ROOT_DIR = '/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/'
folder_calib = os.path.join(ROOT_DIR, 'outputs/Calibration_files')
folder_pyr_mask = os.path.join(ROOT_DIR, 'outputs/3s_pyr_mask')
folder_transformation_matrices = os.path.join(ROOT_DIR, 'outputs/Transformation_matrices')


#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)
slm = dao_setup.slm

# Initialize Cameras
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp

#%% Creating and Displaying a Circular Pupil on the SLM

data_slm = compute_data_slm()
slm.set_data(data_slm)

print('Pupil successfully created on the SLM.')


#%% Capturing a Reference Image

reference_image = camera_wfs.get_data()

# Display the Reference Image
plt.figure()
plt.imshow(reference_image, cmap='gray')
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Display the Focal pane image
fp_image = camera_fp.get_data()
plt.figure()
plt.imshow(fp_image) 
plt.colorbar()
plt.title('PSF')
plt.show()


#%% Creating a Flux Filtering Mask with modulation

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
flux_cutoff = 0.65 # Intensity cutoff threshold - 30%
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

masked_reference_image = reference_image * mask
normalized_reference_image = masked_reference_image / np.abs(np.sum(masked_reference_image))
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Masked Reference Image (DM modulation)')
plt.savefig(os.path.join(folder_pyr_mask, f'masked_ref_image_dm_modulation.png'))
plt.show()
#%% Creating a Flux Filtering Mask with DM Random

summed_image = create_flux_filtering_mask_trial(n_iter=100, verbose=True, verbose_plot=True)

summed_image = np.asarray(summed_image)   

mean_val = np.mean(summed_image)
std_val = np.std(summed_image)

# Display the  Summed Image
plt.figure()
plt.title('Summed Image (DM random)')
plt.imshow(summed_image, vmin=mean_val - 10*std_val, vmax=mean_val + 10*std_val, cmap='viridis')
plt.colorbar()
plt.savefig(os.path.join(folder_pyr_mask, 'summed_image_dm_random_pushpull.png'))
plt.show()

#%%
# Create a filtering mask
flux_cutoff =  0.3#0.64 # Intensity cutoff threshold - 30%
mask = np.zeros(summed_image.shape, dtype=bool)
flux_limit_upper = summed_image.max() * flux_cutoff
mask[summed_image >= flux_limit_upper] = True
print('Flux filtering mask successfully created.')

# Display the Mask
plt.figure()
plt.title('Flux Filtering Mask (DM random)')
plt.imshow(mask, cmap='viridis')
plt.colorbar()
#plt.savefig(os.path.join(folder_pyr_mask, f'mask_dm_random.png'))
plt.show()

# Apply Mask and Crop Results
masked_summed_image = mask*summed_image #[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3]]

# Display the Masked Summed Image
plt.figure()
plt.title('Masked Summed Image (DM random)')
plt.imshow(masked_summed_image, cmap='viridis')
plt.colorbar()
#plt.savefig(os.path.join(folder_pyr_mask, f'masked_summed_image_dm_random.png'))
plt.show()

# Save the Masked Image and Mask
#fits.writeto(os.path.join(folder, f'binned_masked_pyr_images_pup_{pupil_size}mm_3s_pyr.fits'), np.asarray(masked_summed_image), overwrite=True)
#fits.writeto(os.path.join(folder, f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'), np.asarray(mask.astype(np.uint8)), overwrite=True)

masked_reference_image = reference_image * mask
normalized_reference_image = masked_reference_image / np.abs(np.sum(masked_reference_image))
plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.title('Masked Reference Image (DM random)')
#plt.savefig(os.path.join(folder_pyr_mask, f'masked_ref_image_dm_random.png'))
plt.show()