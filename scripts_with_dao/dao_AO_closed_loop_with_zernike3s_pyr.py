#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:41:03 2025

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
import sys
import scipy
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
import matplotlib.animation as animation
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Output folders
folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'
folder_closed_loop_tests = ROOT_DIR / 'outputs/Closed_loop_tests'
folder_turbulence = ROOT_DIR / 'outputs/Phase_screens'

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


#%% Create a deformable mirror (DM)

# Number of actuators
nact = 17

# Create a deformable mirror (DM)
t0 = time.time()
dm_modes = make_gaussian_influence_functions(small_pupil_grid, nact, pupil_size / (nact - 1), crosstalk=0.3)
deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
t1 = time.time()
print(f"Time to create DM: {t1 - t0:.4f} s")
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
deformable_mirror.actuators.fill(1)
plt.figure()
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface')
plt.show()

#%% Create transformation matrices

# Define folder path
nmodes_zernike = 200
Act2Phs, Phs2Act = compute_Act2Phs(nact, small_pupil_grid_Npix, pupil_size, small_pupil_grid, verbose=True)
Znk2Phs, Phs2Znk = compute_Znk2Phs(nmodes_zernike, small_pupil_grid_Npix, pupil_size, small_pupil_grid, verbose=True)
Act2Znk, Znk2Act = compute_Znk2Act(nact, nmodes_zernike, Act2Phs, Phs2Act, Znk2Phs, Phs2Znk, verbose=True)

# Extend Zernike basis for the SLM
Znk2Phs_extended = np.zeros((nmodes_zernike, dataHeight, dataWidth), dtype=np.float32)
Znk2Phs_extended[:, offset_height:offset_height + small_pupil_grid_Npix, offset_width:offset_width + small_pupil_grid_Npix] = Znk2Phs.reshape(nmodes_zernike, small_pupil_grid_Npix, small_pupil_grid_Npix)

# Plot Znk modes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Phs[mode].reshape(small_pupil_grid_Npix, small_pupil_grid_Npix), cmap='viridis')
    axes_flat[i].set_title(f'Znk mode {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Plot Znk projections on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(Znk2Act[mode].reshape(nact, nact), cmap='viridis')
    axes_flat[i].set_title(f' Znk proj. Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()


#%% Load Bias Image, Calibration Mask and Response Matrix

# Load the bias image
bias_filename = f'binned_bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'binned_response_matrix_Znk2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_Znk2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1

#SVD
# Compute SVD
U, S, Vt = np.linalg.svd(IM_Znk2PyWFS)
S_matrix = np.diag(S)
plt.figure()
plt.imshow(S_matrix)
plt.colorbar()
plt.show()

RM_PyWFS2Znk = np.linalg.pinv(IM_Znk2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2Znk.shape}")

#%%

#tip tilt check

# Diffraction limited PSF
camera_fp.Open()
diffraction_limited_psf = pylonGrab(camera_fp, 1)
diffraction_limited_psf /= diffraction_limited_psf.sum()
fp_img_shape = diffraction_limited_psf.shape
print('PSF shape:', fp_img_shape)

# Create the PSF mask 
psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)

print(psf_center)


#%% Capturing a Reference Image and PSF

# Display Pupil Data on SLM
slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))

# Capture a reference image using the WFS camera.
camera_wfs.Open() # Open the Wavefront Sensor (WFS) Camera
time.sleep(0.3)  # Wait for stabilization of SLM
reference_image = pylonGrab(camera_wfs, 10)
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Save the reference image to a FITS file
filename = f'binned_ref_img_pup_{pupil_size}mm_3s_pyr.fits'
fits.writeto(os.path.join(folder_calib, filename), np.asarray(reference_image), overwrite=True)

# Diffraction limited PSF
camera_fp.Open()
diffraction_limited_psf = pylonGrab(camera_fp, 1)
diffraction_limited_psf /= diffraction_limited_psf.sum()
fp_img_shape = diffraction_limited_psf.shape
print('PSF shape:', fp_img_shape)

#Radial Profile 
plt.figure()
plt.plot(diffraction_limited_psf[:,291:311:2])
plt.plot(diffraction_limited_psf[286:316:2,:].T)
plt.show()

# Create the PSF mask 
psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)

plt.figure()
plt.imshow(psf_mask)
plt.colorbar()
plt.title('PSF Mask to compute strehl')
plt.show()

# Integrate the flux in that small region
integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()
print('sum center PSF =', integrated_diff_psf)

# Plot PSF with selected region overlayed
plt.figure()
plt.imshow(diffraction_limited_psf, norm=LogNorm(), cmap='viridis')
plt.colorbar(label='Intensity')
plt.title('PSF with Selected Region')
circle = plt.Circle((psf_center[1], psf_center[0]), radius=50, color='red', fill=False, linewidth=2) # Overlay the integration region (circle)
plt.gca().add_patch(circle)
plt.show()

# Compute the Strehl ratio
#strehl_ratio = integrated_obs / integrated_diff

#%% Defining the number of Znk modes used for Closed lop simulations

# Define Znk modes to consider
nmodes_Znk = 150
Znk2Phs_new = Znk2Phs[:nmodes_Znk, :]
Phs2Znk_new = scipy.linalg.pinv(Znk2Phs_new)
Znk2Act_new = Znk2Act[:nmodes_Znk, :]
Act2Znk_new = scipy.linalg.pinv(Znk2Act_new)
IM_Znk2PyWFS_new = IM_Znk2PyWFS[:nmodes_Znk, :]
RM_PyWFS2Znk_new = np.linalg.pinv(IM_Znk2PyWFS_new, rcond=0.10)

#%% AO loop with a fixed Znk mode 

# Select Znk mode and amplitude
mode = 1
amp = 5

data_Znk = Znk2Phs_new[mode].reshape(small_pupil_grid_Npix, small_pupil_grid_Npix) * amp

# Main loop parameters
num_iterations = 1
gain = 1

anim_path = folder_closed_loop_tests
anim_name= f'closed_loop_test_Znk_mode_{mode}_amp_{amp}.gif'
anim_title= f'Znk mode {mode} amp {amp}'

#Stop grapping
camera_wfs.StopGrabbing()

closed_loop_simulation(num_iterations, gain, data_Znk, anim_path, anim_name, anim_title,
                           RM_PyWFS2Znk_new, Znk2Act_new, Act2Znk_new, Phs2Znk_new, 
                           deformable_mirror, slm, camera_wfs, camera_fp, 
                           small_pupil_grid_Npix, data_pupil, data_pupil_outer, data_pupil_inner, 
                           pupil_mask, small_pupil_mask, mask, bias_image)

# %% AO loop with Phase screens
plt.close('all')

# Load the FITS data
# Load the phase screen
wl= 1700
pup = 1.52
seeing = 1.0
loopspeed = 1.0
#filename = f'phase_screen_cube_phase_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz.fits'
filename = f'turbulence_cube_phase_seeing_2arcsec_L_40m_tau0_5ms_lambda_500nm_pup_1.52m_1.0kHz.fits'
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename))
hdu = hdul[0]
fits_data = hdu.data[1000:2000, :, :]

# Initialize the phase screen array to hold all frames
num_frames = fits_data.shape[0]
data_phase_screen = np.zeros((num_frames, small_pupil_grid_Npix, small_pupil_grid_Npix), dtype=np.float32)
data_phase_screen = fits_data

# scale the phase screen to given seeing and wavelength
data_phase_screen = data_phase_screen*small_pupil_mask*(500/wl)*((seeing/2)**(5/6))
    
# Main loop parameters
num_iterations = 100
gain =  1

anim_path = folder_closed_loop_tests
anim_name= f'closed_loop_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz_gain_{gain}.gif'
anim_title= f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'

#Stop grapping
camera_wfs.StopGrabbing()

closed_loop_simulation(num_iterations, gain, data_phase_screen/2, anim_path, anim_name, anim_title,
                           RM_PyWFS2Znk_new, Znk2Act_new, Act2Znk_new, Phs2Znk_new, 
                           deformable_mirror, slm, camera_wfs, camera_fp, 
                           small_pupil_grid_Npix, data_pupil, data_pupil_outer, data_pupil_inner, 
                           pupil_mask, small_pupil_mask, mask, bias_image)


# %% Load Phase screens from Nicolas
plt.close('all')

# Load the phase screen
wl= 1700
pup = 1.5
seeing = 1
loopspeed = 1.0
filename = 'turbOpd_seeing500_2.00_wind_5.0_Dtel_1.5.fits'
hdul = fits.open(os.path.join(folder_turbulence / 'Papyrus', filename))
hdu = hdul[0]
fits_data = hdu.data[0:1000, :, :]
num_frames = fits_data.shape[0]

# Create a new 550x550xN array filled with zeros
data_phase_screen = np.zeros((num_frames, 550, 550))

# Place the 500x500xN data_phase_screen in the center of the new array
for i in range(num_frames):
    data_phase_screen[i, 25:525, 25:525] = fits_data[i, :, :]

# scale the phase screen to given seeing and wavelength
data_phase_screen = data_phase_screen*small_pupil_mask*(500/wl)*((seeing/2)**(5/6))

# Main loop parameters
num_iterations = 1000
gain =  1# Fixed gain value

anim_path = folder_closed_loop_tests
anim_name= f'closed_loop_sturbOpd_seeing500_{seeing}_wind_5.0_Dtel_1.5_lambda_{wl}nm.gif'
anim_title= f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'

#Stop grapping
camera_wfs.StopGrabbing()

closed_loop_simulation(num_iterations, gain, data_phase_screen, anim_path, anim_name, anim_title,
                           RM_PyWFS2Znk_new, Znk2Act_new, Act2Znk_new, Phs2Znk_new, 
                           deformable_mirror, slm, camera_wfs, camera_fp, 
                           small_pupil_grid_Npix, data_pupil, data_pupil_outer, data_pupil_inner, 
                           pupil_mask, small_pupil_mask, mask, bias_image)


#%%


