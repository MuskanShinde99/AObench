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
from DEVICES_3.Basler_Pylon.test_pylon import * # import library for camera functions

# Set the working directory
os.chdir('/home/laboptic/Documents/optlab-master/PROJECTS_3/RISTRETTO/Banc AO')

# Import specific modules
import src.dao_setup  # Import the setup file

#%%

# Access the SLM
slm = dao_setup.slm

#Access the cameras
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp
camera_wfs.Open()

#Get cropping size for making the image square
#img = pylonGrab(camera, 1)
#height, width = img.shape[:2]
#img_size = min(height, width)
#camera.Close()

#%%

#Create a circular pupil on SLM
data_pupil = dao_setup.data_pupil

# New TTF amplitudes for Tip, Tilt, and Focus
# new_ttf_amplitudes = [-0.3, 0.1, 0.4]  # New values for tip, tilt, and focus amplitudes

# Update the pupil with the new values
# data_pupil = dao_setup.update_pupil(new_ttf_amplitudes=new_ttf_amplitudes)
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
