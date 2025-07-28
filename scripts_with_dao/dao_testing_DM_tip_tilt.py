#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 01:55:52 2025

@author: ristretto-dao
"""

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
from astropy.io import fits
import os
import sys
import scipy
from pathlib import Path
from datetime import datetime
from src.config import config

ROOT_DIR = config.root_dir

# Import Specific Modules
import dao
from src.dao_setup import init_setup, set_data_dm
setup = init_setup()  # Import all variables from setup
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import * 
from src.psf_centring_algorithm_functions import *
from src.scan_modes_functions import *
from src.ao_loop_functions import *


#%% Creating and Displaying a Circular Pupil on the SLM

# Compute and display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)

#%% Load transformation matrices

# # Load transformation matrices from shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

# From folder 
KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_KL}_nact_{nact}.fits'))
KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{nmodes_KL}_npupil_{npix_small_pupil_grid}.fits'))


# plt.figure()
# plt.imshow(KL2Act[3,:].reshape(nact,nact))
# plt.colorbar()
# plt.title('KL mode')
# plt.show()

# plt.figure()
# plt.imshow(small_pupil_mask*KL2Phs[0,:].reshape(npix_small_pupil_grid,npix_small_pupil_grid))
# plt.colorbar()
# plt.title('KL mode')
# plt.show()

#%% testing tip tilt
y0 = None  # Reference y-coordinate

for amp in np.arange(0, 15, 1):
    kl_mode = amp * KL2Act[1]
    data_dm, data_slm = set_data_dm(
        kl_mode,
        setup=setup,
    )

    img = camera_fp.get_data()

    # Get coordinates of maximum
    max_coords = np.unravel_index(np.argmax(img), img.shape)
    y, x = max_coords

    if y0 is None:
        y0 = y  # Set reference y at amp = 0

    dy = y - y0
    print(f"amp: {amp}, 19*amp: {19 * amp}, difference from amp=0: {dy}")

# Set to flat
data_slm = compute_data_slm()
slm.set_data(data_slm)
time.sleep(wait_time)
