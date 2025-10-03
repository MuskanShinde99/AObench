#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:29 2025

@author: laboptic
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
dm_phase_shm = shm.dm_phase_shm
phase_screen_shm = shm.phase_screen_shm
phase_residuals_shm = shm.phase_residuals_shm
residual_modes_shm = shm.residual_modes_shm
slopes_image_shm = shm.slopes_image_shm
normalized_psf_shm = shm.normalized_psf_shm
computed_modes_shm = shm.computed_modes_shm
commands_shm = shm.commands_shm
strehl_ratio_shm = shm.strehl_ratio_shm
norm_flux_pyr_img_shm = shm.norm_flux_pyr_img_shm

dm_kl_modes_shm = shm.dm_kl_modes_shm
dm_act_shm = shm.dm_act_shm
dm_flat_papy_shm = shm.dm_flat_papy_shm
KL2Act_papy_shm = shm.KL2Act_papy_shm
dm_papy_shm = shm.dm_papy_shm

#Loading folder
folder_calib = config.folder_calib
folder_gui = setup.folder_gui

# Load hardware and configuration parameters
camera_wfs = setup.camera_wfs
camera_fp = setup.camera_fp
npix_small_pupil_grid = setup.npix_small_pupil_grid


#%% Creating and Displaying a Circular Pupil on the SLM

#DM set to flat
set_data_dm(setup=setup)
#dm_flat_papy_shm.set_data(setup.dm_flat.astype(np.float32))
#fits.writeto(folder_calib / 'dm_flat_papy.fits', setup.dm_flat.astype(np.float32), overwrite=True)


#%% Load transformation matrices

KL2Act_papy = KL2Act_papy_shm.get_data().T

plt.figure()
plt.imshow(KL2Act_papy[177, :])
plt.colorbar()
plt.show()