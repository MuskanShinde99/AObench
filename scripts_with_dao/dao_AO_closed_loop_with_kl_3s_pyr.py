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
import matplotlib.animation as animation
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
from DEVICES_3.Basler_Pylon.test_pylon import *
import dao
from src.create_circular_pupil import *
from src.tilt import *
from src.utils import *
from src.calibration_functions import *
import src.dao_setup as dao_setup  # Import the setup file
from src.kl_basis_eigenmodes import computeEigenModes, computeEigenModes_notsquarepupil
from src.create_transformation_matrices import *
from src.ao_loop import *

folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'
folder_closed_loop_tests = ROOT_DIR / 'outputs/Closed_loop_tests'
folder_turbulence = ROOT_DIR / 'outputs/Phase_screens'

#%% Accessing Devices

# Initialize Spatial Light Modulator (SLM)
slm = dao_setup.slm

# Initialize Cameras
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp

#%% Creating and Displaying a Circular Pupil on the SLM

# Access the pupil data from the setup file
pupil_size = dao_setup.pupil_size
npix_small_pupil_grid = dao_setup.npix_small_pupil_grid
small_pupil_mask = dao_setup.small_pupil_mask
pupil_mask = dao_setup.pupil_mask

# Display Pupil Data on SLM
data_slm = compute_data_slm()
slm.set_data(data_slm)

#%% Create a deformable mirror (DM)

# Number of actuators
nact = dao_setup.nact
nact_valid = dao_setup.nact_valid
dm_modes = dao_setup.dm_modes

deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
print('DM created')
print("Number of DM modes =", nmodes_dm)

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
# deformable_mirror.actuators.fill(1)
# plt.figure()
# plt.imshow(deformable_mirror.surface.shaped)
# plt.colorbar()
# plt.title('Deformable Mirror Surface')
# plt.show()

#%% Load transformation matrices

# Define folder path

nmodes_kl = nact_valid
KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{nmodes_kl}_npupil_{npix_small_pupil_grid}.fits'))
Phs2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Phs2KL_npupil_{npix_small_pupil_grid}_nkl_{nmodes_kl}.fits'))

KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'))
Act2KL = fits.getdata(os.path.join(folder_transformation_matrices, f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'))

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
IM_filename = f'binned_response_matrix_KL2Act_push-pull_pup_{pupil_size}mm_nact_{nact}_amp_0.1_3s_pyr.fits'
IM_KL2PyWFS = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1

#SVD
# Compute SVD
U, S, Vt = np.linalg.svd(IM_KL2PyWFS)
S_matrix = np.diag(S)
# plt.figure()
# plt.imshow(S_matrix)
# plt.colorbar()
# plt.show()

RM_PyWFS2KL = np.linalg.pinv(IM_KL2PyWFS, rcond=0.10)
print(f"Shape of the response matrix: {RM_PyWFS2KL.shape}")

#%% Capturing a Reference Image and PSF

# Display Pupil Data on SLM
slm.set_data(data_slm)

# Capture a reference image using the WFS camera.
time.sleep(dao_setup.wait_time)  # Wait for stabilization of SLM
reference_image = camera_wfs.get_data()
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
diffraction_limited_psf = camera_fp.get_data()
#diffraction_limited_psf /= diffraction_limited_psf.sum()
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

#%% Defining the number of Kl modes used for Closed lop simulations

# Define KL modes to consider
nmodes_kl = 100
KL2Phs_new = KL2Phs[:nmodes_kl, :]
Phs2KL_new = scipy.linalg.pinv(KL2Phs_new)
KL2Act_new = KL2Act[:nmodes_kl, :]
Act2KL_new = scipy.linalg.pinv(KL2Act_new)
IM_KL2PyWFS_new = IM_KL2PyWFS[:nmodes_kl, :]
RM_PyWFS2KL_new = np.linalg.pinv(IM_KL2PyWFS_new, rcond=0.10)

#%% AO loop with a fixed KL mode 
plt.close('all')

# Select KL mode and amplitude
mode = 0
amp = 1
data_kl = KL2Phs_new[mode].reshape(npix_small_pupil_grid, npix_small_pupil_grid) * amp * small_pupil_mask

# Main loop parameters
num_iterations = 10
gain = 1
leakage = 0
delay=0

anim_path= folder_closed_loop_tests
anim_name= f'closed_loop_test_KL_mode_{mode}_amp_{amp}.gif'
anim_title= f'KL mode {mode} amp {amp}'

closed_loop_test(num_iterations, gain, leakage, delay, data_kl, anim_path, anim_name, anim_title,
                           RM_PyWFS2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new, mask, bias_image)

# %% AO loop with Phase screens
plt.close('all')

deformable_mirror.flatten() 

# Load the FITS data
seeing = 1.0# in arcsec
wl= 1700  #in nm

pup = 1.52 #in m
wl_ref = 500 #in nm
seeing_ref = 2.0 # in arcsec
loopspeed = 1.0 # in KHz

#filename = f'phase_screen_cube_phase_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz.fits'
filename = f'Papyrus/turbulence_cube_phase_seeing_2arcsec_L_40m_tau0_5ms_lambda_500nm_pup_1.52m_1.0kHz.fits'
hdul = fits.open(os.path.join(folder_turbulence, filename))
hdu = hdul[0]
fits_data = hdu.data[0:501, :, :]

# Initialize the phase screen array to hold all frames
num_frames = fits_data.shape[0]
data_phase_screen = np.zeros((num_frames, npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
data_phase_screen = fits_data

# scale the phase screen to given seeing and wavelength
data_phase_screen = data_phase_screen*small_pupil_mask*(wl_ref/wl)*((seeing/seeing_ref)**(5/6)) #in radians
data_phase_screen = data_phase_screen / (2*np.pi) # in Waves
    
# Main loop parameters
num_iterations = 100
gain = 1
leakage = 0
delay=0

anim_path = folder_closed_loop_tests / 'Papyrus'
anim_name= f'AO_bench_closed_loop_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz_gain_{gain}_iterations_{num_iterations}.gif'
anim_title= f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'


strehl_ratios, residual_phases = closed_loop_test(num_iterations, gain, leakage, delay, data_phase_screen, anim_path, anim_name, anim_title,
                           RM_PyWFS2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new, mask, bias_image)

# save strehl ratio and phase residual arrays
strehl_ratios_path = os.path.join(anim_path, f"strehl_ratios_{anim_name.replace('.gif', '.npy')}")
residual_phases_path = os.path.join(anim_path, f"residual_phases_{anim_name.replace('.gif', '.npy')}")
np.save(strehl_ratios_path, np.array(strehl_ratios))
np.save(residual_phases_path, np.array(residual_phases))
    
# figure strehl ratios and rphase residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(strehl_ratios, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Strehl Ratio')

plt.subplot(1, 2, 2)
plt.plot(residual_phases, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Residual Phase [2π rad]')
plt.title(f'AO Bench -- {anim_title}')

plt.tight_layout()
plt.show()

#%% For different seeing in a loop

# Define parameters
seeing_values = [1.0, 2.0, 3.0, 4.0]  # Different seeing values
wl = 1700  # in nm
pup = 1.52  # in m
wl_ref = 500  # in nm
seeing_ref = 2.0  # in arcsec
loopspeed = 1.0  # in KHz


# Iterate over each seeing value
for seeing in seeing_values:
    plt.close('all')

    # Load the FITS data
    filename = f'Papyrus/turbulence_cube_phase_seeing_2arcsec_L_40m_tau0_5ms_lambda_500nm_pup_1.52m_1.0kHz.fits'
    hdul = fits.open(os.path.join(folder_turbulence, filename))
    hdu = hdul[0]
    fits_data = hdu.data[0:510, :, :]

    # Initialize the phase screen array to hold all frames
    num_frames = fits_data.shape[0]
    data_phase_screen = np.zeros((num_frames, npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
    data_phase_screen = fits_data

    # Scale the phase screen to the given seeing and wavelength
    data_phase_screen = data_phase_screen * small_pupil_mask * (wl_ref / wl) * ((seeing / seeing_ref) ** (5 / 6))  # in radians
    data_phase_screen = data_phase_screen / (2 * np.pi)  # in Waves

    # Main loop parameters
    num_iterations = 500
    gain = 1

    anim_path = os.path.join(folder_closed_loop_tests, 'Papyrus') 
    anim_name = f'AO_bench_closed_loop_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz_gain_{gain}_iterations_{num_iterations}.gif'
    anim_title = f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'

    # Stop grapping
    camera_wfs.StopGrabbing()

    strehl_ratios, residual_phases = closed_loop_test(
        num_iterations, gain, leakage, delay, data_phase_screen, anim_path, anim_name, anim_title,
        RM_PyWFS2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new,
        deformable_mirror, slm, camera_wfs, camera_fp,
        npix_small_pupil_grid, data_pupil, data_pupil_outer, data_pupil_inner,
        pupil_mask, small_pupil_mask, mask, bias_image
    )

    # Save Strehl ratio and phase residual arrays
    strehl_ratios_path = os.path.join(anim_path, f"strehl_ratios_{anim_name.replace('.gif', '.npy')}")
    residual_phases_path = os.path.join(anim_path, f"residual_phases_{anim_name.replace('.gif', '.npy')}")
    np.save(strehl_ratios_path, np.array(strehl_ratios))
    np.save(residual_phases_path, np.array(residual_phases))

    # Figure Strehl ratios and residual phases
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(strehl_ratios, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Strehl Ratio')

    plt.subplot(1, 2, 2)
    plt.plot(residual_phases, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Phase [2π rad]')
    plt.title(f'AO Bench -- {anim_title}')

    plt.tight_layout()
    plt.show()



# %% Load Phase screens from Nicolas
plt.close('all')

# Load the phase screen
wl= 1700
pup = 1.5
seeing = 2
loopspeed = 1.0

filename = 'Papyrus/turbOpd_seeing500_2.00_wind_5.0_Dtel_1.5.fits'
hdul = fits.open(os.path.join(folder_turbulence, filename))
hdu = hdul[0]
fits_data = hdu.data[0:10, :, :]
num_frames = fits_data.shape[0]

# Create a new 550x550xN array filled with zeros
data_phase_screen = np.zeros((num_frames, 550, 550))

# Place the 500x500xN data_phase_screen in the center of the new array
for i in range(num_frames):
    data_phase_screen[i, 25:525, 25:525] = fits_data[i, :, :]

# scale the phase screen to given seeing and wavelength
data_phase_screen = data_phase_screen*small_pupil_mask*(500/wl)*((seeing/2)**(5/6))

# Main loop parameters
num_iterations = 1
gain =  1# Fixed gain value

anim_path= os.path.join(folder_closed_loop_tests, 'Papyrus') 
anim_name= f'closed_loop_sturbOpd_seeing500_{seeing}_wind_5.0_Dtel_1.5_lambda_{wl}nm.gif'
anim_title= f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'

#Stop grapping
camera_wfs.StopGrabbing()

closed_loop_test(num_iterations, gain, leakage, delay, data_phase_screen, anim_path, anim_name, anim_title,
                           RM_PyWFS2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new, 
                           deformable_mirror, slm, camera_wfs, camera_fp, 
                           npix_small_pupil_grid, data_pupil, data_pupil_outer, data_pupil_inner, 
                           pupil_mask, small_pupil_mask, mask, bias_image)


#%%


