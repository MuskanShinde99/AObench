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

# Turn off laser

# Turn off laser
if las is not None:
    las.enable(0)
    time.sleep(2)  # Allow some time for laser to turn off
    print("The laser is OFF")
    
else:
    input("Turn OFF the laser and press Enter to continue")
    
# Take a bias image

# Capture and average 1000 bias frames
n_frames=1000
bias_image = np.median([camera_wfs.get_data() for i in range(n_frames)], axis=0)
bias_image_shm.set_data(bias_image)

# Plot
plt.figure()
plt.imshow(bias_image, cmap='gray')
plt.title('Bias image')
plt.colorbar()
plt.show()

# Save the Bias Image

fits.writeto(os.path.join(folder_calib, f'bias_image.fits'), np.asarray(bias_image), overwrite=True)

# Set bias image to zero for PAPY SIM tests
#bias_image=np.zeros_like(bias_image) #TODO: Remove it

# Turn on laser

if las is not None:
    las.enable(1)
    time.sleep(2)
    print("The laser is ON")
    
else:
    input("Turn ON the laser and press Enter to continue")
