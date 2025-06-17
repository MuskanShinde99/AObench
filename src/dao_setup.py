#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:56:42 2025

@author: laboptic
"""

# Import libraries
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import *
import time
from astropy.io import fits
import os
import dao
from skimage.transform import resize  

# Set the working directory
os.chdir('/home/ristretto-dao/optlab-master')
from DEVICES_3.Basler_Pylon.test_pylon import *
from DEVICES_3.Thorlabs.MCLS1 import mcls1

# Set the working directory
os.chdir('/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/')

# Import specific modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *

ROOT_DIR = '/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/'
folder_calib = os.path.join(ROOT_DIR, 'outputs/Calibration_files')
folder_pyr_mask = os.path.join(ROOT_DIR, 'outputs/3s_pyr_mask')
folder_transformation_matrices = os.path.join(ROOT_DIR, 'outputs/Transformation_matrices')
floder_closed_loop_tests = os.path.join(ROOT_DIR, 'outputs/Closed_loop_tests')
folder_turbulence = os.path.join(ROOT_DIR, 'outputs/Phase_screens')
#%% Start the laser

channel = 1
las = mcls1("/dev/ttyUSB0")
las.set_channel(channel)
las.enable(1) # Turn on laser
las.set_current(49) #55mA is a good value for pyramid images
print('Laser is ON')
  
#%% Configuration Camera

# To set camera
camera_wfs = dao.shm('/tmp/cam1.im.shm')
camera_fp = dao.shm('/tmp/cam2.im.shm')

fps_wfs = dao.shm('/tmp/cam1Fps.im.shm')
fps_wfs.set_data(fps_wfs.get_data()*0+300)

fps_fp = dao.shm('/tmp/cam2Fps.im.shm')
fps_fp.set_data(fps_fp.get_data()*0+20)

img = camera_wfs.get_data()
img_size = img.shape[0]
# # To get camera image
# camera_wfs.get_data()
# camera_fp.get_data()

#%% Configuration SLM

# Initializes the SLM library
slm=dao.shm('/tmp/slm.im.shm')
print('SLM is open')

# Get SLM dimensions
dataWidth, dataHeight = slm.get_data().shape[1], slm.get_data().shape[0]
pixel_size = 8e-3  # Pixel size in mm

wait_time = 0.3

# dataWidth = slm.width_px
# dataHeight = slm.height_px
# pixel_size = slm.pixelsize_m * 1e3  # Pixel size in mm

#%% Create Pupil

# Parameters for the circular pupil
pupil_size = 4  # [mm]
npix_pupil = int(pupil_size / pixel_size)   # Convert pupil size to pixels
blaze_period_outer = 20
blaze_period_inner = 15
tilt_amp_outer = 150
tilt_amp_inner = -70.5  # -70.5 -67 -40

# Create the circular pupil mask
pupil_grid = make_pupil_grid([dataWidth, dataHeight], [dataWidth * pixel_size, dataHeight * pixel_size])
vlt_aperture_generator = make_obstructed_circular_aperture(pupil_size, 0, 0, 0)
pupil_mask = evaluate_supersampled(vlt_aperture_generator, pupil_grid, 1)
pupil_mask =pupil_mask.reshape(dataHeight, dataWidth)
pupil_mask = pupil_mask.astype(bool)

# Create circular pupil
data_pupil = create_slm_circular_pupil(tilt_amp_outer, tilt_amp_inner, pupil_size, pupil_mask, slm)

# Create Zernike basis
zernike_basis = make_zernike_basis(4, pupil_size, pupil_grid)
zernike_basis = [mode / np.ptp(mode) for mode in zernike_basis]
zernike_basis = np.asarray(zernike_basis)

# Create a Tip, Tilt, and Focus (TTF) matrix with specified amplitudes as the diagonal elements
ttf_amplitudes = [-0.010094774185274713, 0.15822279475032697, 0.4]  # Tip, Tilt, and Focus amplitudes - Focus 0.4
ttf_amplitude_matrix = np.diag(ttf_amplitudes)
ttf_matrix = ttf_amplitude_matrix @ zernike_basis[1:4, :]  # Select modes 1 (tip), 2 (tilt), and 3 (focus)

data_ttf = np.zeros((dataHeight, dataWidth), dtype=np.float32)
data_ttf[:, :] = (ttf_matrix[0] + ttf_matrix[1] + ttf_matrix[2]).reshape(dataHeight, dataWidth) 

data_pupil = data_pupil + data_ttf # Add TTF matrix to pupil


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
    data_pupil = create_slm_circular_pupil(tilt_amp_outer, tilt_amp_inner, pupil_size, pupil_mask, slm)

    # Create a new Tip, Tilt, and Focus (TTF) matrix with the updated amplitudes
    ttf_matrix = ttf_amplitudes @ zernike_basis[1:4, :]  # Select modes 1 (tip), 2 (tilt), and 3 (focus)
    
    # Create TTF data
    data_ttf = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_ttf[:, :] = (ttf_matrix[0] + ttf_matrix[1] + ttf_matrix[2]).reshape(dataHeight, dataWidth)

    # Add the new TTF to the pupil and return
    return data_pupil + data_ttf

#%% Create a pupil grid for a smaller pupil area

# Set up pupil grid dimensions with size 1.1 times the pupil size
oversizing = 1.1
npix_small_pupil_grid = int(npix_pupil * oversizing) 
small_pupil_grid = make_pupil_grid(npix_small_pupil_grid, npix_small_pupil_grid * pixel_size)
# print('New  small pupil grid created')
# print('Pupil grid shape:', npix_small_pupil_grid, npix_small_pupil_grid)

# Calculate offsets to center the pupil grid with respect to the SLM grid
offset_height = (dataHeight - npix_small_pupil_grid) // 2
offset_width = (dataWidth - npix_small_pupil_grid) // 2

# Create a grid mask 
small_pupil_grid_mask = np.zeros((dataHeight, dataWidth), dtype=bool)
small_pupil_grid_mask[offset_height:offset_height + npix_small_pupil_grid, offset_width:offset_width + npix_small_pupil_grid] = 1

# Create the circular pupil mask for small square grid
small_pupil_mask = pupil_mask[offset_height:offset_height + npix_small_pupil_grid, 
                               offset_width:offset_width + npix_small_pupil_grid]
# plt.figure()
# plt.imshow(small_pupil_mask)
# plt.colorbar()
# plt.title('Small Pupil Mask')
# plt.show()

#%% Create outer and inner data slm

# Create a new `data_pupil_outer` with the same size as `data_pupil`
data_pupil_outer = np.copy(data_pupil)
data_pupil_outer[pupil_mask] = 0  # Zero out inner region given by 'pupil_mask'

# Create a new `data_pupil_inner` with the same size as `small_pupil_mask`
data_pupil_inner = np.copy(data_pupil[offset_height:offset_height + npix_small_pupil_grid, 
                              offset_width:offset_width + npix_small_pupil_grid])
data_pupil_inner[~small_pupil_mask] = 0 # Zero out inner region outside by 'small_pupil_mask'

# plt.figure()
# plt.imshow(data_pupil_outer)
# plt.colorbar()
# plt.title('Data Pupil Outer')
# plt.show()

# plt.figure()
# plt.imshow(data_pupil_inner)
# plt.colorbar()
# plt.title('Data Pupil Inner')
# plt.show()

#%% Define deformable mirror

# Number of actuators
nact = 17

# Create influence functions for deformable mirror (DM)
t0 = time.time()
dm_modes_full = make_gaussian_influence_functions(small_pupil_grid, nact, pupil_size / (nact - 1), crosstalk=0.3)

# Resize the pupil mask to exactly nactxnact and compute the valid actuqator indices
valid_actuators_mask = resize(np.logical_not(small_pupil_mask), (nact, nact), order=0, anti_aliasing=False, preserve_range=True).astype(int)
valid_actuator_indices= np.column_stack(np.where(valid_actuators_mask))

# Compute valid actuator counts
nact_total = valid_actuators_mask.size
nact_outside = np.sum(valid_actuators_mask)
nact_valid = nact_total - nact_outside

#Put the dm modes outside the valid actuator indices to zero
dm_modes = np.asarray(dm_modes_full)

for x, y in valid_actuator_indices:
    dm_modes[x * nact + y] = 0  # Zero out the corresponding mode

dm_modes = ModeBasis(dm_modes.T, small_pupil_grid)

#Create shared memory for DM
dm_act_shm = dao.shm('/tmp/dm_act.im.shm', np.zeros((npix_small_pupil_grid, npix_small_pupil_grid)).astype(np.float32)) 

deformable_mirror = DeformableMirror(dm_modes)