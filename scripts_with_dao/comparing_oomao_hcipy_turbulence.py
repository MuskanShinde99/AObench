#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Libraries
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
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
import matplotlib.animation as animation
from pathlib import Path
from src.config import config

ROOT_DIR = config.root_dir

# Import Specific Modules
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.calibration_functions import *
from src.dao_setup import *  # Import all variables from setup
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *
from src.ao_loop import *

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)

# Initialize Cameras


#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file



# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))
print('Pupil successfully created on the SLM.')

#%% Create transformation matrices

# Define folder path
KL2Phs, Phs2KL = compute_KL2Phs(nact, small_pupil_grid_Npix, pupil_size, small_pupil_grid, small_pupil_mask, verbose=True)

#%%

# Load first turbulence data
filename = 'turbulence_cube_phase_seeing_2arcsec_L_40m_tau0_5ms_lambda_500nm_pup_1.52m_1.0kHz.fits'
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename))
hdu = hdul[0]
turb_hcipy = hdu.data[0, :, :]
KL_modes_turb_hcipy = turb_hcipy.flatten() @ Phs2KL

# Load second turbulence data
filename = 'turbOpd_seeing500_2.00_wind_5.0_Dtel_1.5.fits'
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename))
hdu = hdul[0]
turb_oomao = np.zeros((550, 550))
turb_oomao[25:525, 25:525] = hdu.data[0, :, :]
KL_modes_turb_oomao = turb_oomao.flatten() @ Phs2KL

# Plot both in subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(KL_modes_turb_hcipy)
axs[0].set_title('Turbulence HCIPy')
axs[0].set_xlabel('KL modes')
axs[0].set_ylabel('Phase [rad]')

axs[1].plot(KL_modes_turb_oomao)
axs[1].set_title('Turbulence OOMAO')
axs[1].set_xlabel('KL modes')
axs[1].set_ylabel('Phase [rad]')

plt.tight_layout()
plt.show()

#%%
# Load turbulence data
filename_hcipy = 'turbulence_cube_phase_seeing_2arcsec_L_40m_tau0_5ms_lambda_500nm_pup_1.52m_1.0kHz_cube2.fits'
filename_oomao = 'turbOpd_seeing500_2.00_wind_5.0_Dtel_1.5.fits'

# Number of frames to load
num_frames = 500

# Load HCIPy turbulence
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename_hcipy))
hdu = hdul[0]
turb_hcipy = hdu.data[:num_frames, :, :]  
hdul.close()

# Compute KL modes
KL_modes_hcipy = np.array([frame.flatten() @ Phs2KL for frame in turb_hcipy])

# Load OOMAO turbulence
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename_oomao))
hdu = hdul[0]
turb_oomao = np.zeros((num_frames, 550, 550))

for i in range(num_frames):
    turb_oomao[i, 25:525, 25:525] = hdu.data[i, :, :]  

hdul.close()

# Compute KL modes
KL_modes_oomao = np.array([frame.flatten() @ Phs2KL for frame in turb_oomao])

# Compute mean and standard deviation
mean_KL_hcipy = np.mean(KL_modes_hcipy, axis=0)
std_KL_hcipy = np.std(KL_modes_hcipy, axis=0)

mean_KL_oomao = np.mean(KL_modes_oomao, axis=0)
std_KL_oomao = np.std(KL_modes_oomao, axis=0)

# Plot standard deviation
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(std_KL_hcipy, color='r')
axs[0].set_title('Standard Deviation of KL Modes - HCIPy')
axs[0].set_xlabel('KL modes')
axs[0].set_ylabel('Phase [rad]')

axs[1].plot(std_KL_oomao, color='r')
axs[1].set_title('Standard Deviation of KL Modes - OOMAO')
axs[1].set_xlabel('KL modes')
axs[1].set_ylabel('Phase [rad]')

plt.tight_layout()
plt.show()


