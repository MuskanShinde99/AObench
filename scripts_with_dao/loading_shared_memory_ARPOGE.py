#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:50:12 2025

@author: ristretto-dao
"""

# Standard library
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from matplotlib.colors import LogNorm
from datetime import datetime

# Import Specific Modules
from src.config import config
from src.dao_setup import init_setup
from src.utils import *
from src.shm_loader import shm

# Flag to identify on-sky data
OnSky = False
suffix = "_OnSky" if OnSky else ""

#Loading setup
setup = init_setup()
setup = reload_setup()

#Loading shared memories
npix_valid_shm = shm.npix_valid_shm
reference_image_shm = shm.reference_image_shm
normalized_ref_image_shm = shm.normalized_ref_image_shm 
valid_pixels_mask_shm = shm.valid_pixels_mask_shm
#Loading folder
folder_calib = config.folder_calib
folder_gui = setup.folder_gui
folder_ARPOGE = '/home/daouser/RISTRETTO/ARPOGE/data'

# Load hardware and configuration parameters
camera_wfs = setup.camera_wfs
camera_fp = setup.camera_fp
npix_small_pupil_grid = setup.npix_small_pupil_grid


#%% Load Calibration Mask and Interaction Matrix and Normalized Reference image to the shared memory 

# Load the calibration mask 
mask_filename = f'mask_3s_pyr{suffix}.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")
fits.writeto(os.path.join(folder_ARPOGE, 'mask.fits'), mask, overwrite=True)

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape[0]
shm.npix_valid_shm.set_data(np.array([[npix_valid]]))

# Load the response matrix 
IM_filename = f'processed_response_cube_KL2PWFS_push-pull_nact_17_amp_0.05_3s_pyr.fits'
IM_KL2S_full = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1
IM_KL2S = compute_response_matrix(IM_KL2S_full, mask).astype(np.float32)
RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
RM_S2KL = RM_S2KL.T
print(f"Shape of the reconstruction matrix: {RM_S2KL.shape}")

# Load reference image
reference_image = fits.getdata(folder_calib / 'reference_image_raw.fits')
normalized_reference_image = normalize_image(mask, mask)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

plt.figure()
plt.imshow(normalized_reference_image)
plt.colorbar()
plt.show()

#Load in shared memories
# mask_arpoge_shm = dao.shm('/tmp/mask.shm')
# mask_arpoge_shm.set_data(mask)

shm.valid_pixels_mask_shm.set_data(mask)
shm.npix_valid_shm.set_data(np.array([[npix_valid]]))
shm.valid_pixels_mask_shm.set_data(mask)
shm.reference_image_shm.set_data(reference_image)
#shm.reference_image_normalized_shm.set_data(normalized_reference_image)
# KL2S_shm = dao.shm('/tmp/KL2S.im.shm', np.zeros((setup.nmodes_KL, npix_valid), dtype=np.float64))
# KL2S_shm.set_data(np.asanyarray(response_matrix_filtered).astype(np.float64))
# S2KL_shm = dao.shm('/tmp/S2KL.im.shm', np.zeros((setup.nmodes_KL, npix_valid), dtype=np.float64))
# S2KL_shm.set_data(np.asanyarray(RM_S2KL).astype(np.float64))

#Save to ARPOGE

fits.writeto(os.path.join(folder_ARPOGE, 'reference_image_normalized.fits') , normalized_reference_image, overwrite=True)
fits.writeto(os.path.join(folder_ARPOGE, 'RM_S2KL.fits'), RM_S2KL, overwrite=True)
