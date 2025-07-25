#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:16:49 2025

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
import sys
from pathlib import Path
from src.config import config

ROOT_DIR = config.root_dir
folder_slm_tests = config.root_dir / 'outputs/SLM_tests'
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao

# Import Specific Modules
from src.dao_setup import init_setup
setup = init_setup()  # Import all variables from setup
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.create_transformation_matrices import *

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)

# Initialize Cameras

#%% Creating and Displaying a Circular Pupil on the SLM

data_pupil = ((data_pupil * 256) % 256).astype(np.uint8)

plt.figure()
plt.imshow(data_pupil, cmap='gray')
plt.colorbar()
plt.title('Data Pupil')
plt.show()

# Display Pupil Data on SLM
#slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

# Access the pupil data from the setup file
data_zeros = np.zeros((dataHeight, dataWidth), dtype=np.uint8)

plt.figure()
plt.imshow(data_zeros, cmap='gray')
plt.colorbar()
plt.title('Data Zeros')
plt.show()

#Size of image from the camera
img = pylonGrab(camera_wfs, 1)
height, width = img.shape[:2]

#%%
# Parameters
Nframes = 50
iterations = 1000

data_cube = np.zeros((Nframes, height, width))

# Prepare the plot
plt.figure(figsize=(10, 6))
plt.title('SLM Response Time')
plt.xlabel('Time [s]')
plt.ylabel('Mean Flux')

camera_wfs.StopGrabbing()
camera_wfs.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

tgrab_ = np.zeros((Nframes))
tslm = np.zeros((iterations))

# Loop for iterations
for i in range(iterations):
    #print(i)
    
    # Determine data to display
    if i % 2 == 0:
        data_slm = data_pupil
    else:
        data_slm = data_zeros

    t0 = time.time()
    # Show data on SLM
    slm.set_data(data_slm)
    t1 = time.time()
    tslm[i] = t1 - t0

    # Grab data cube from camera
    #data_cube = pylonGrab_datacube(camera, Nframes)  # Replace with your camera data acquisition method
    for j in range(Nframes):
        grabResult = camera_wfs.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            data_cube[j,:,:] = np.array(grabResult.Array.astype("float64"))
        grabResult.Release()
        tgrab_[j] = time.time() - t1
            
    # Compute mean flux
    mean_values = np.mean(data_cube, axis=(1,2))
    
    # Plot mean flux
    plt.plot(tgrab_, mean_values)

# Finalize the plot
plt.tight_layout()
plt.savefig(os.path.join(folder_slm_tests, f'slm_response_time_test_iterations_{iterations}.png'))
plt.show()

plt.figure(), plt.plot(tslm)
plt.xlabel('Iterations')
plt.ylabel('Time [s]')
plt.title('SLM latency')
plt.savefig(os.path.join(folder_slm_tests, f'slm_latency_iterations_{iterations}.png'))
plt.show()
camera_wfs.StopGrabbing()


