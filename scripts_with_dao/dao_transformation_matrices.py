#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 09:28:12 2025

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
from hcipy import *
import dao
import sys
from pathlib import Path

# Configure root paths 
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from DEVICES_3.Basler_Pylon.test_pylon import *

#import src.dao_setup as dao_setup  # Import the setup file
from src.dao_setup import *
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.dao_create_flux_filtering_mask import *
from src.psf_centring_algorithm import *
from src.create_transformation_matrices import *
#from src.create_shared_memories import *

#%% Create transformation matrices

Act2Phs, Phs2Act = compute_Act2Phs(nact, npix_small_pupil_grid, dm_modes, folder_transformation_matrices, verbose=True)

# Create KL modes
nmodes_kl = nact_valid
Act2KL, KL2Act = compute_KL2Act(nact, npix_small_pupil_grid, nmodes_kl, dm_modes, small_pupil_mask, folder_transformation_matrices, verbose=True)
KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nmodes_kl, Act2Phs, Phs2Act, KL2Act, Act2KL, folder_transformation_matrices, verbose=True)
# KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nact_valid , dm_modes, small_pupil_mask, Act2Phs, folder_transformation_matrices, verbose=True)
# Act2KL, KL2Act = compute_KL2Act(nact, nact_valid, Act2Phs, Phs2Act, KL2Phs, Phs2KL, folder_transformation_matrices, verbose=True)

# Plot KL projections on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Act[mode].reshape(nact, nact), cmap='viridis')
    axes_flat[i].set_title(f' KL2Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Plot KL modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Phs[mode].reshape(npix_small_pupil_grid, npix_small_pupil_grid)* small_pupil_mask, cmap='viridis')
    axes_flat[i].set_title(f'KL2Phs {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Create Zernike modes
nmodes_zernike = 200
Znk2Phs, Phs2Znk = compute_Znk2Phs(nmodes_zernike, npix_small_pupil_grid, pupil_size, small_pupil_grid, folder_transformation_matrices, verbose=True)
Act2Znk, Znk2Act = compute_Znk2Act(nact, nmodes_zernike, Act2Phs, Phs2Act, Znk2Phs, Phs2Znk, folder_transformation_matrices, verbose=True)

# Plot KL modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Phs[mode].reshape(npix_small_pupil_grid, npix_small_pupil_grid), cmap='viridis')
    axes_flat[i].set_title(f'Znk2Phs {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Plot KL projections on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Act[mode].reshape(nact, nact), cmap='viridis')
    axes_flat[i].set_title(f' Znk2Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Act2Phs_shm.set_data(Act2Phs)
# Phs2Act_shm.set_data(Phs2Act)

# KL2Act_shm.set_data(KL2Act)
# Act2KL_shm.set_data(Act2KL)
# KL2Phs_shm.set_data(KL2Phs)
# Phs2KL_shm.set_data(Phs2KL)

# Znk2Act_shm.set_data(Znk2Act)
# Act2Znk_shm.set_data(Act2Znk)
# Znk2Phs_shm.set_data(Znk2Phs)
# Phs2Znk_shm.set_data(Phs2Znk)
