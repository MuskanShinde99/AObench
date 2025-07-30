#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:29 2025

@author: laboptic
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from matplotlib.colors import LogNorm
from src.utils import *
from collections import deque
from src.dao_setup import init_setup
from src.utils import set_dm_actuators, set_data_dm
from .shm_loader import shm

setup = init_setup()
dm_phase_shm = shm.dm_phase_shm
phase_screen_shm = shm.phase_screen_shm
phase_residuals_shm = shm.phase_residuals_shm
residual_modes_shm = shm.residual_modes_shm
slopes_image_shm = shm.slopes_image_shm
normalized_psf_shm = shm.normalized_psf_shm
computed_modes_shm = shm.computed_modes_shm
commands_shm = shm.commands_shm
dm_kl_modes_shm = shm.dm_kl_modes_shm
dm_act_shm = shm.dm_act_shm


# Load hardware and configuration parameters
camera_wfs = kwargs.get("camera_wfs", setup.camera_wfs)
camera_fp = kwargs.get("camera_fp", setup.camera_fp)
npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", setup.npix_small_pupil_grid)

# Load the folder
folder_gui = kwargs.get("folder_gui", setup.folder_gui)

# Flatten the DM
set_data_dm(setup=setup)
#deformable_mirror.flatten()

# Reference image 
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
if verbose:
    print('Reference image shape:', pyr_img_shape)

# Diffraction limited PSF
diffraction_limited_psf = diffraction_limited_psf.astype(np.float32)
diffraction_limited_psf /= diffraction_limited_psf.sum()
fp_img_shape = diffraction_limited_psf.shape
if verbose:
    print('PSF shape:', fp_img_shape)

# Create the PSF mask 
psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)
# Integrate the flux in that small region
integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()
if verbose:
    print('sum center PSF =', integrated_diff_psf)

# Get valid pixel indices from the mask
valid_pixels_indices = np.where(mask > 0)

    
# Initialize arrays to store Strehl ratio and total residual phase
strehl_ratios = np.zeros(num_iterations)
residual_phase_stds = np.zeros(num_iterations)

start_time = time.time()

i = 0
while True:
    # Capture and process WFS image
    slopes_image = get_slopes_image(
        mask,
        bias_image,
        normalized_reference_image,
        setup=setup,
    )
    slopes_image_shm.set_data(slopes_image)
    slopes = slopes_image[valid_pixels_indices].flatten()
    #fits.writeto(os.path.join(folder_gui, f'slopes_image.fits'), slopes_image, overwrite=True)
    
    # Compute KL modes present
    computed_modes = slopes @ RM_S2KL 
    # multiply by two because this mode is computed for DM surface and we want DM phase
    computed_modes_shm.set_data(computed_modes) # setting shared memory
    
    # Compute actuator commands
    act_pos = computed_modes @ KL2Act
    commands_shm.set_data(act_pos) # setting shared memory

    # Capture PSF
    fp_img = camera_fp.get_data()
    fp_img = np.maximum(fp_img, 1e-10)
    normalized_psf_shm.set_data((fp_img / np.max(fp_img))) # setting shared memory
    #fits.writeto(os.path.join(folder_gui, f'normalized_psf.fits'), (fp_img / np.max(fp_img)), overwrite=True)

    # Compute Strehl ratio
    observed_psf = fp_img / fp_img.sum()
    integrated_obs_psf = observed_psf[psf_mask].sum()
    strehl_ratio = integrated_obs_psf / integrated_diff_psf
    # strehl_ratio = np.max(observed_psf) / np.max(diffraction_limited_psf)
    strehl_ratios[i] = strehl_ratio

    i += 1  # increment loop index


