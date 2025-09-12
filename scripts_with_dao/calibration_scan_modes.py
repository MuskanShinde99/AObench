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

# Flag to identify on-sky data
OnSky = False
suffix = "_OnSky" if OnSky else ""

#Loading setup
setup = init_setup()
setup = reload_setup()

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

#%% Load mask

# Load the calibration mask for processing images.
mask_filename = f'mask_3s_pyr{suffix}.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

 #%% Scanning modes to find zero of the pyramid

test_values = np.arange(-0.5, 0.5, 0.05)
mode_index = 3 # 0 - focus, 1 - astimgatism, 2 -astigmatism 
#scan_othermode_amplitudes(test_values, mode_index, update_setup_file=True)
scan_othermode_amplitudes_wfs_std(test_values, mode_index, mask, 
                                  update_setup_file=False)
  
#revise the crieteria to standard deviation of intensities within the valid pixels