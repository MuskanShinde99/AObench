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

#Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Compute and display Pupil Data on SLM
set_data_dm(setup=setup)

# Capure the Reference Image
n_frames=1000
reference_image = (np.mean([camera_wfs.get_data().astype(np.float32) for i in range(n_frames)], axis=0)).astype(camera_wfs.get_data().dtype)
#reference_image = fits.getdata(os.path.join(folder_calib, f'mask_3s_pyr.fits'))
# average over several frames
fits.writeto(folder_calib / 'reference_image_raw.fits', reference_image, overwrite=True)
fits.writeto(folder_calib / f'reference_image_raw_{timestamp}.fits', reference_image, overwrite=True)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()
