#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:52:29 2025

@author: laboptic
"""

from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import *
import time
from astropy.io import fits
import os
import sys
from DEVICES_3.Basler_Pylon.test_pylon import * # import library for camera functions
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import specific modules
from src.dao_setup import *  # Import all variables from setup

#%%

# Access the SLM

#Access the cameras
camera_wfs.Open()

#Get cropping size for making the image square
#img = pylonGrab(camera, 1)
#height, width = img.shape[:2]
#img_size = min(height, width)
#camera.Close()

#%%

#Create a circular pupil on SLM

# New TTF amplitudes for Tip, Tilt, and Focus
# new_ttf_amplitudes = [-0.3, 0.1, 0.4]  # New values for tip, tilt, and focus amplitudes

# Update the pupil with the new values
# data_pupil = update_pupil(new_ttf_amplitudes=new_ttf_amplitudes)
print('Pupil created on the SLM')

# Display the pupil
plt.figure()
plt.imshow(data_pupil)
plt.title(f'Data pupil')
plt.colorbar()
plt.show()

#Show data on SLM:
slm.set_data(((data_pupil*256)%256).astype(np.uint8))


#%% 

#Capture image
img = pylonGrab(camera_wfs, 1)
plt.figure()
plt.imshow(img)
plt.colorbar()
plt.show()

#%%
