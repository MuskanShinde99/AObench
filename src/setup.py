# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:50:31 2024

@author: RISTRETTO
"""
# Import libraries
# from holoeye import detect_heds_module_path, slmdisplaysdk
# from holoeye.showSLMPreview import showSLMPreview
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import *
import time
from astropy.io import fits
import os
import sys
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import specific modules
from create_circular_pupil import *
from tilt import *
from utils import *
from DEVICES_3.Basler_Pylon.test_pylon import pylonGrab, pylonGrab_datacube, pylonGrabSingle
from DEVICES_3.Thorlabs.MCLS1 import mcls1

#%% Start the laser

channel = 1
las = mcls1("COM5")
las.set_channel(channel)
las.enable(1)
las.set_current(50) #55mA is a good value for pyramid images
print('Laser is ON')
  
#%% Configuration Camera

# Iinitialize the camera
devices = pylon.TlFactory.GetInstance().EnumerateDevices() # Get a list of all connected devices
first_camera = devices[0] # Access the first camera
second_camera = devices[1] # Access the second camera

# Create an InstantCamera object for the second camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(second_camera))
camera_fp  =pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(first_camera))
print('Both cameras accessible: camera and camera_fp')

#Set the camera parameters 
camera.Open()
camera.PixelFormat.SetValue('Mono10')
texp = 100  # Exposure time in microseconds
camera.ExposureTime.SetValue(texp)
maxfps = 1e6/camera.SensorReadoutTime()
fps = camera.AcquisitionFrameRate()
#camera.Close()

# camera.AcquisitionFrameRate()
# camera.SensorReadoutTime()

#%% Camera ROI setup

# # set the camera size
# new_width = camera.Width.GetValue() - camera.Width.GetInc()
# if new_width >= camera.Width.GetMin():
#     camera.Width.SetValue(new_width)
    
# # Set the camera  width, height, and offsets
# column_size = 608
# row_size = 496
# column_offset = 1600
# row_offset = 1416
# camera.Width.SetValue(column_size)
# camera.Height.SetValue(row_size)
# camera.OffsetX.SetValue(column_offset) # not working give max and min offset that can be set as 0
# camera.OffsetY.SetValue(row_offset)
    

# # # Select auto function ROI 1
# camera.AutoFunctionROISelector.SetValue("ROI1")
# camera.AutoFunctionROIOffsetX.SetValue(100)  # Horizontal offset of ROI
# camera.AutoFunctionROIOffsetY.SetValue(100)  # Vertical offset of ROI
# camera.AutoFunctionROIWidth.SetValue(500)    # Width of ROI
# camera.AutoFunctionROIHeight.SetValue(476)   # Height of ROI

camera.AutoFunctionROISelector.SetValue("ROI1")  # Ensure the correct ROI is selected
camera.AutoFunctionROIOffsetX.SetValue(0)
camera.AutoFunctionROIOffsetY.SetValue(0)
camera.AutoFunctionROIWidth.SetValue(camera.Width.GetMax())
camera.AutoFunctionROIHeight.SetValue(camera.Height.GetMax())

#%% Configuration SLm

# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()
error = slm.open() # Detect SLMs and open a window on the selected SLM
print('SLM is open')

# Open the SLM preview window in "Fit" mode:
#showSLMPreview(slm, scale=0.0)

# Get SLM dimensions
dataWidth = slm.width_px
dataHeight = slm.height_px
pixel_size = slm.pixelsize_m * 1e3  # Pixel size in mm

#%% Create Pupil

# Parameters for the circular pupil
pupil_size = 4  # [mm]
blaze_period_outer = 20
blaze_period_inner = 15
tilt_amp_outer = 150
tilt_amp_inner = -40  # -25
Npix_pupil = int(pupil_size / pixel_size)   # Convert pupil size to pixels
pupil_grid = make_pupil_grid([dataWidth, dataHeight], [dataWidth * pixel_size, dataHeight * pixel_size])

# Create circular pupil
data_pupil = create_slm_circular_pupil_new(tilt_amp_outer, tilt_amp_inner, pupil_size, slm)

# Create Zernike basis
zernike_basis = make_zernike_basis(4, pupil_size, pupil_grid)
zernike_basis = [mode / np.ptp(mode) for mode in zernike_basis]
zernike_basis = np.asarray(zernike_basis)

# Create a Tip, Tilt, and Focus (TTF) matrix with specified amplitudes as the diagonal elements
ttf_amplitudes = np.diag([0.20, 0.45, 0.4])  # Tip, Tilt, and Focus amplitudes
ttf_matrix = ttf_amplitudes @ zernike_basis[1:4, :]  # Select modes 1 (tip), 2 (tilt), and 3 (focus)

data_ttf = np.zeros((dataHeight, dataWidth), dtype=np.float32)
data_ttf[:, :] = (ttf_matrix[0] + ttf_matrix[1] + ttf_matrix[2]).reshape(dataHeight, dataWidth) 

data_pupil = data_pupil + data_ttf

# Create the circular pupil mask:
x = np.linspace(-dataWidth / 2, dataWidth / 2, dataWidth)
y = np.linspace(-dataHeight / 2, dataHeight / 2, dataHeight)
xx, yy = np.meshgrid(x, y)
rr = np.abs(xx + 1j * yy)
pupil_mask = rr < Npix_pupil / 2# Add TTF matrix to pupil

# Function to update pupil with new TTF amplitude
def update_pupil(new_ttf_amplitudes=None, new_tilt_amp_outer=None, new_tilt_amp_inner=None):
    global ttf_amplitudes, tilt_amp_outer, tilt_amp_inner, data_pupil

    # Update the amplitudes and tilt parameters if new values are provided
    if new_ttf_amplitudes is not None:
        ttf_amplitudes = np.diag(new_ttf_amplitudes)

    if new_tilt_amp_outer is not None:
        tilt_amp_outer = new_tilt_amp_outer

    if new_tilt_amp_inner is not None:
        tilt_amp_inner = new_tilt_amp_inner

    # Recalculate pupil with updated values
    data_pupil = create_slm_circular_pupil_new(tilt_amp_outer, tilt_amp_inner, pupil_size, slm)

    # Create a new Tip, Tilt, and Focus (TTF) matrix with the updated amplitudes
    ttf_matrix = ttf_amplitudes @ zernike_basis[1:4, :]  # Select modes 1 (tip), 2 (tilt), and 3 (focus)
    
    # Create TTF data
    data_ttf = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_ttf[:, :] = (ttf_matrix[0] + ttf_matrix[1] + ttf_matrix[2]).reshape(dataHeight, dataWidth)

    # Add the new TTF to the pupil and return
    return data_pupil + data_ttf

