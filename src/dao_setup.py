#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:56:42 2025

@author: laboptic
"""

# Import Libraries
from hcipy import *
from matplotlib.colors import LogNorm
import gc
from tqdm import tqdm
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
from astropy.io import fits
import os
import sys
import scipy
import matplotlib.animation as animation
from pathlib import Path
from skimage.transform import resize

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from DEVICES_3.Thorlabs.MCLS1 import mcls1
import dao
from src.create_circular_pupil import *
from src.dao_create_flux_filtering_mask import *
from src.tilt import *
from src.utils import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *


folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'
folder_closed_loop_tests = ROOT_DIR / 'outputs/Closed_loop_tests'
folder_turbulence = ROOT_DIR / 'outputs/Phase_screens'
folder_gui = ROOT_DIR / 'outputs/GUI_tests'

#%% Start the laser

channel = 1
las = mcls1("/dev/ttyUSB0")
las.set_channel(channel)
#las.enable(1) # 1 to turn on laser, 0 to turn off
las.set_current(49) #55mA is a good value for pyramid images
print('Laser is Accessible')
  
#%% Configuration Camera

# To set camera
camera_wfs = dao.shm('/tmp/cam1.im.shm')
camera_fp = dao.shm('/tmp/cam2.im.shm')

fps_wfs = dao.shm('/tmp/cam1Fps.im.shm')
fps_wfs.set_data(fps_wfs.get_data()*0+300)

fps_fp = dao.shm('/tmp/cam2Fps.im.shm')
fps_fp.set_data(fps_fp.get_data()*0+20)

img = camera_wfs.get_data()
img_size_wfs_cam = img.shape[0]

img_fp = camera_fp.get_data()
img_size_fp_cam = img_fp.shape[0]

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

wait_time = 0.15

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
zernike_basis = make_zernike_basis(11, pupil_size, pupil_grid)
zernike_basis = [mode / np.ptp(mode) for mode in zernike_basis]
zernike_basis = np.asarray(zernike_basis)

# [-0.0813878287964559, 0.09992195172893337]
# Create a Tip-Tilt (TT) matrix with specified amplitudes as the diagonal elements
tt_amplitudes = [-1.680357716565717, 0.12651517048626593]  # Tip and Tilt amplitudes
tt_amplitude_matrix = np.diag(tt_amplitudes)
tt_matrix = tt_amplitude_matrix @ zernike_basis[1:3, :]  # Select modes 1 (tip) and 2 (tilt)

data_tt = np.zeros((dataHeight, dataWidth), dtype=np.float32)
data_tt[:, :] = (tt_matrix[0] + tt_matrix[1]).reshape(dataHeight, dataWidth)

othermodes_amplitudes = [0.36500000000000077, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Focus (mode 3) + modes 4 to 10
othermodes_amplitude_matrix = np.diag(othermodes_amplitudes)
othermodes_matrix = othermodes_amplitude_matrix @ zernike_basis[3:11, :]  # Select modes 3 (focus) to 10

data_othermodes = np.zeros((dataHeight, dataWidth), dtype=np.float32)
data_othermodes[:, :] = (othermodes_matrix[0] + othermodes_matrix[1] + othermodes_matrix[2]).reshape(dataHeight, dataWidth) 

# Add all the modes to data pupil
data_pupil = data_pupil + data_tt + data_othermodes  # Add TT and higher-order terms to pupil


# Function to update pupil with new TT amplitudes and other modes
def update_pupil(new_tt_amplitudes=None, new_othermodes_amplitudes=None,
                 new_tilt_amp_outer=None, new_tilt_amp_inner=None):
    """Recompute the pupil using updated tip/tilt and higher-order amplitudes.

    Parameters
    ----------
    new_tt_amplitudes : sequence of float, optional
        Two-element list ``[tip, tilt]`` for the TT matrix.
    new_othermodes_amplitudes : sequence of float, optional
        Amplitudes for focus and higher-order Zernike modes
        (e.g., ``[focus, astig_x, astig_y, ...]``).
    new_tilt_amp_outer : float, optional
        Outer grating tilt amplitude.
    new_tilt_amp_inner : float, optional
        Inner grating tilt amplitude.
    """
    global tt_amplitudes, othermodes_amplitudes
    global tilt_amp_outer, tilt_amp_inner, data_pupil

    # Update the amplitudes and tilt parameters if new values are provided
    if new_tt_amplitudes is not None:
        tt_amplitudes = list(new_tt_amplitudes)

    if new_othermodes_amplitudes is not None:
        othermodes_amplitudes = list(new_othermodes_amplitudes)

    if new_tilt_amp_outer is not None:
        tilt_amp_outer = new_tilt_amp_outer

    if new_tilt_amp_inner is not None:
        tilt_amp_inner = new_tilt_amp_inner

    # Recalculate pupil with updated values
    data_pupil = create_slm_circular_pupil(tilt_amp_outer, tilt_amp_inner,
                                           pupil_size, pupil_mask, slm)

    # Create a new Tip-Tilt (TT) matrix with the updated amplitudes
    tt_matrix = np.diag(tt_amplitudes) @ zernike_basis[1:3, :]  # Select modes 1 (tip) and 2 (tilt)
    
    # Create TT data
    data_tt = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_tt[:, :] = (tt_matrix[0] + tt_matrix[1]).reshape(dataHeight, dataWidth)

    # Recompute focus and higher-order terms
    othermodes_matrix = np.diag(othermodes_amplitudes) @ zernike_basis[3:11, :]
    data_othermodes = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_othermodes[:, :] = (othermodes_matrix[0] + othermodes_matrix[1] + othermodes_matrix[2]).reshape(dataHeight, dataWidth)

    # Add the new TT and other modes to the pupil and return
    return data_pupil + data_tt + data_othermodes

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

deformable_mirror = DeformableMirror(dm_modes_full)

nmodes_dm = deformable_mirror.num_actuators

# print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
# deformable_mirror.actuators.fill(1)
# plt.figure()
# plt.imshow(deformable_mirror.opd.shaped)
# plt.colorbar()
# plt.title('Deformable Mirror Surface OPD')
# plt.show()

#%% Create shared memory

nmodes_dm = nact_valid
nmodes_KL = nact_valid
nmodes_Znk = nact_valid

