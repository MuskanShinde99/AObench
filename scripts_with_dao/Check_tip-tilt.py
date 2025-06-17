#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:31:23 2025

@author: laboptic
"""

# Import Libraries
from matplotlib.colors import LogNorm
import gc
from tqdm import tqdm
from pypylon import pylon
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from hcipy import *
import time
from astropy.io import fits
import os
import scipy
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
import matplotlib.animation as animation

# Set the Working Directory
os.chdir('/home/laboptic/Documents/optlab-master/PROJECTS_3/RISTRETTO/Banc AO')

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
import src.dao_setup as dao_setup  # Import the setup file
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *
from src.ao_loop import *

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)
slm = dao_setup.slm

# Initialize Cameras
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp


#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file
dataHeight = dao_setup.dataHeight
dataWidth = dao_setup.dataWidth

data_pupil = dao_setup.data_pupil
data_pupil_inner = dao_setup.data_pupil_inner
data_pupil_outer = dao_setup.data_pupil_outer
pupil_size = dao_setup.pupil_size
pupil_grid = dao_setup.pupil_grid
pupil_mask = dao_setup.pupil_mask

small_pupil_grid_Npix = dao_setup.small_pupil_grid_Npix
small_pupil_grid = dao_setup.small_pupil_grid
small_pupil_grid_mask = dao_setup.small_pupil_grid_mask
small_pupil_mask = dao_setup.small_pupil_mask
offset_height = dao_setup.offset_height #offset to center the small pupil grid on the full slm screen
offset_width = dao_setup.offset_width  #offset to center the small pupil grid on the full slm screen


#%% Create transformation matrices

# Define folder path
folder = '/home/laboptic/Documents/RISTRETTO_AO_bench/Transformation_matrices'

nmodes_zernike = 200
Znk2Phs, Phs2Znk = compute_Znk2Phs(nmodes_zernike, small_pupil_grid_Npix, pupil_size, small_pupil_grid, verbose=True)

# Extend Zernike basis for the SLM
Znk2Phs_extended = np.zeros((nmodes_zernike, dataHeight, dataWidth), dtype=np.float32)
Znk2Phs_extended[:, offset_height:offset_height + small_pupil_grid_Npix, offset_width:offset_width + small_pupil_grid_Npix] = Znk2Phs.reshape(nmodes_zernike, small_pupil_grid_Npix, small_pupil_grid_Npix)


#%% 

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))
print('Pupil successfully created on the SLM.')

# Diffraction limited PSF
time.sleep(0.3)  # Wait for stabilization of SLM
camera_fp.Open()
diffraction_limited_psf = pylonGrab(camera_fp, 1)
diffraction_limited_psf /= diffraction_limited_psf.max()
fp_img_shape = diffraction_limited_psf.shape
print('PSF shape:', fp_img_shape)

plt.figure()
plt.imshow(diffraction_limited_psf)
plt.colorbar()
plt.title('PSF')
plt.show()

#Radial Profile 
plt.figure()
plt.plot(diffraction_limited_psf[:,254:274])
#plt.plot(diffraction_limited_psf[271:291,:].T)
plt.axhline(0.5, linestyle='--', color='r', label='x=0.5')
plt.show()

#%% Check shift for 1 lambda/D
from scipy.ndimage import center_of_mass

for amp in range(5):  # Iterates over 0, 1, 2
    print(f"Amp: {amp}")  # Print current amplitude for clarity

    # Display Pupil Data on SLM
    mode = 1
    data_zernike = amp * Znk2Phs_extended[mode] 
    slm.set_data((((data_pupil + data_zernike) * 256) % 256).astype(np.uint8))

    # PSF
    time.sleep(0.3)  # Wait for stabilization of SLM
    psf = pylonGrab(camera_fp, 1)
    
    # Compute Center of Mass (COM)
    psf_center = center_of_mass(psf)  # (y, x) coordinates


    # Create the PSF mask
    #psf_mask, psf_center = create_psf_mask(psf, crop_size=100, radius=50)

    # Plot PSF with center coordinates
    plt.figure()
    plt.imshow(psf)  # Use 'inferno' colormap for better contrast
    plt.colorbar(label="Normalized Intensity")
    plt.scatter(psf_center[1], psf_center[0], color='cyan', marker='+', s=100, label="Center")
    plt.title(f"PSF Tilt = {amp} lambda/D, Center: {psf_center}")
    plt.legend()
    plt.show()
    
#%% Defocus

from scipy.ndimage import center_of_mass

for amp in np.arange(-0.05, 0.05, 0.01):  # Iterates over 0, 1, 2
    print(f"Amp: {amp}")  # Print current amplitude for clarity

    # Display Pupil Data on SLM
    mode = 3
    data_zernike = amp * Znk2Phs_extended[mode]
    slm.set_data((((data_pupil + data_zernike) * 256) % 256).astype(np.uint8))

    # PSF
    time.sleep(0.3)  # Wait for stabilization  of SLM
    psf = pylonGrab(camera_fp, 1)
    
    # print max
    print('Man intensity:', psf.max())

#%%
mode = 3
amp = 0.4
data_zernike = amp * Znk2Phs_extended[mode]
slm.set_data((((data_pupil + data_zernike) * 256) % 256).astype(np.uint8))

time.sleep(0.3)
img = pylonGrab(camera_wfs, 1)

plt.figure()
plt.imshow(img)
plt.show()