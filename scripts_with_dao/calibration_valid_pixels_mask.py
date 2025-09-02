#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:20 2025

@author: laboptic
"""

# Standard library
import os
import time
from datetime import datetime
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import importlib

# Import Specific Modules
import dao
from src.dao_setup import init_setup, las
from src.utils import set_data_dm, reload_setup

from src.config import config
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import * 
from src.psf_centring_algorithm_functions import *
from src.shm_loader import shm
from src.scan_modes_functions import *
from src.ao_loop_functions import *

#Loading setup
setup = init_setup()

#Loading folder
folder_calib = config.folder_calib

#Loading shared memories
bias_image_shm = shm.bias_image_shm
valid_pixels_mask_shm = shm.valid_pixels_mask_shm
npix_valid_shm = shm.npix_valid_shm
reference_image_shm = shm.reference_image_shm
normalized_ref_image_shm = shm.normalized_ref_image_shm
reference_psf_shm = shm.reference_psf_shm
KL2Act_papy_shm = shm.KL2Act_papy_shm
dm_flat_papy_shm = shm.dm_flat_papy_shm
dm_papy_shm = shm.dm_papy_shm


#%% Setting DM to flat

# Compute and display Pupil Data on SLM
# data_slm = compute_data_slm()
# slm.set_data(data_slm)
# time.sleep(wait_time)

# DM set to flat
set_data_dm(setup=setup)
dm_flat_papy_shm.set_data(setup.dm_flat.astype(np.float32))
fits.writeto(folder_calib / 'dm_flat_papy.fits', setup.dm_flat.astype(np.float32), overwrite=True)

#%% Load transformation matrices

# # Load transformation matrices from shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

# From folder 
# KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{setup.nmodes_KL}_nact_{setup.nact}.fits'))
# KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{setup.nmodes_KL}_npupil_{setup.npix_small_pupil_grid}.fits'))

KL2Act_papy = KL2Act_papy_shm.get_data().T

# plt.figure()
# plt.plot(KL2Act_papy[1,:])
# plt.show()

#%% Creating a Flux Filtering Mask

method='dm_random'
flux_cutoff = 0.08 # 0.06 - papy dm random; 0.2 - geneva dm random
modulation_angles = np.arange(0, 360, 1)  # angles of modulation
modulation_amp = 15 # in lamda/D
n_iter=500 # number of iternations for dm random commands

mask = create_flux_filtering_mask(method, flux_cutoff, KL2Act_papy[0], KL2Act_papy[1],
                               modulation_angles, modulation_amp, n_iter,
                               create_summed_image=True, verbose=False, verbose_plot=True,
                               OnSky=False,)

valid_pixels_mask_shm.set_data(mask)

# Get valid pixels from the mask
#put with in the function to create mask
valid_pixels_indices = np.where(mask > 0)
npix_valid = valid_pixels_indices[0].shape[0]
npix_valid_shm.set_data(np.array([[npix_valid]]))
print(f'Number of valid pixels = {npix_valid}')

#Reset the DM to flat
set_data_dm(setup=setup)

# Create shared memories that depends on number of valid pixels
KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm' , np.zeros((setup.nmodes_KL, setup.img_size_wfs_cam_x*setup.img_size_wfs_cam_y), dtype=np.float64))
slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1), dtype=np.uint64))
KL2S_shm = dao.shm('/tmp/KL2S.im.shm' , np.zeros((setup.nmodes_KL, npix_valid), dtype=np.float64))
S2KL_shm = dao.shm('/tmp/S2KL.im.shm' , np.zeros((npix_valid, setup.nmodes_KL), dtype=np.float64))
