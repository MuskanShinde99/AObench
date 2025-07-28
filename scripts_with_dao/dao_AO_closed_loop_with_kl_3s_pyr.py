#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:41:03 2025

@author: laboptic
"""

# Import Libraries
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import numpy as np
import time
from astropy.io import fits
import os
import scipy


# Import Specific Modules
from src.config import config
ROOT_DIR = config.root_dir
import dao
from src.dao_setup import init_setup
setup = init_setup()  # Import all variables from setup
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import * 
from src.psf_centring_algorithm_functions import *
from src.shm_loader import shm
num_iterations_shm = shm.num_iterations_shm
gain_shm = shm.gain_shm
leakage_shm = shm.leakage_shm
delay_shm = shm.delay_shm
from src.scan_modes_functions import scan_othermode_amplitudes
from src.ao_loop_functions import *

#%% Creating and Displaying a Circular Pupil on the SLM

# Display Pupil Data on SLM
set_data_dm(setup=setup)

#%% Load transformation matrices

# From folder 
KL2Act = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Act_nkl_{setup.nmodes_KL}_nact_{setup.nact}.fits'))
KL2Phs = fits.getdata(os.path.join(folder_transformation_matrices, f'KL2Phs_nkl_{setup.nmodes_KL}_npupil_{setup.npix_small_pupil_grid}.fits'))

# # From shared memories
# KL2Act = KL2Act_shm.get_data()
# KL2Phs = KL2Phs_shm.get_data()

#%% Load Bias Image, Calibration Mask and Interaction Matrix

# Load the bias image
bias_filename = f'binned_bias_image.fits'
bias_image = fits.getdata(os.path.join(folder_calib, bias_filename))
print(f"Bias image shape: {bias_image.shape}")

# Load the calibration mask for processing images.
mask_filename = f'binned_mask_3s_pyr.fits'
mask = fits.getdata(os.path.join(folder_calib, mask_filename))
print(f"Mask dimensions: {mask.shape}")

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)

# Load the response matrix 
IM_filename = f'binned_response_matrix_KL2S_filtered_nact_{setup.nact}_amp_0.1_3s_pyr.fits'
IM_KL2S = fits.getdata(os.path.join(folder_calib, IM_filename))  # /0.1

RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)
print(f"Shape of the response matrix: {RM_S2KL.shape}")

#%% Load Reference Image and PSF

# Load reference image
time.sleep(wait_time)  # Wait for stabilization of SLM
reference_image = fits.getdata(folder_calib / 'reference_image_raw.fits')
normalized_reference_image = normalize_image(reference_image, mask, bias_image)
pyr_img_shape = reference_image.shape
print('Reference image shape:', pyr_img_shape)

#Plot
plt.figure()
plt.imshow(reference_image)
plt.colorbar()
plt.title('Reference Image')
plt.show()

# Load Diffraction limited PSF
diffraction_limited_psf = fits.getdata(folder_calib / 'reference_psf.fits')
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
nmodes_kl = 175
KL2Phs_new = KL2Phs[:nmodes_kl, :]
Phs2KL_new = scipy.linalg.pinv(KL2Phs_new)
KL2Act_new = KL2Act[:nmodes_kl, :]
Act2KL_new = scipy.linalg.pinv(KL2Act_new)
IM_KL2S_new = IM_KL2S[:nmodes_kl, :]
RM_S2KL_new = np.linalg.pinv(IM_KL2S_new, rcond=0.10)


# %% Loading turbulence
plt.close('all')

setup.deformable_mirror.flatten() 

# Load the FITS data
seeing = 4.0# in arcsec
wl= 1500  #in nm

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
data_phase_screen = np.zeros((num_frames, setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float32)
data_phase_screen = fits_data

# scale the phase screen to given seeing and wavelength
data_phase_screen = data_phase_screen*setup.small_pupil_mask*(wl_ref/wl)*((seeing/seeing_ref)**(5/6)) #in radians
data_phase_screen = data_phase_screen / (2*np.pi) # in Waves
data_phase_screen = data_phase_screen.astype(np.float32)


# have a shared memory for phase screens 

#%% AO loop with turbulence
 
plt.figure()
plt.imshow(data_phase_screen[0,:, :])
plt.show()
   
# Main loop parameters
num_iterations = 100
gain = 1
leakage = 0
delay= 0

# setting shared memories
num_iterations_shm.set_data(np.array([[num_iterations]]))
gain_shm.set_data(np.array([[gain]]))
leakage_shm.set_data(np.array([[leakage]]))
delay_shm.set_data(np.array([[delay_shm]]))

# # loading from shared memory
# num_iterations = int(num_iterations_shm.get_data()[0][0])
# gain = gain_shm.get_data()[0][0]
# leakage = leakage_shm.get_data()[0][0]
# delay = int(leakage_shm.get_data()[0][0])

# defining path for saving plots
anim_path = folder_closed_loop_tests / 'Papyrus'
anim_name= f'AO_bench_closed_loop_seeing_{seeing}arcsec_L_40m_tau0_5ms_lambda_{wl}nm_pup_{pup}m_{loopspeed}kHz_gain_{gain}_iterations_{num_iterations}.gif'
anim_title= f'Seeing: {seeing} arcsec, λ: {wl} nm, Loop speed: {loopspeed} kHz'

#AO loop
strehl_ratios, residual_phases = closed_loop_test(num_iterations, gain, leakage, delay, data_phase_screen, anim_path, anim_name, anim_title,
                           RM_S2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new, mask, bias_image, 
                           reference_image=reference_image, diffraction_limited_psf=diffraction_limited_psf,
                           verbose=True, verbose_plot=False)

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


#%% AO loop with a fixed KL mode 
plt.close('all')

# Select KL mode and amplitude
mode = 0
amp = 1
data_kl = KL2Phs_new[mode].reshape(setup.npix_small_pupil_grid, setup.npix_small_pupil_grid) * amp * setup.small_pupil_mask

# Main loop parameters
num_iterations = 1
gain = 1
leakage = 0
delay=0

anim_path= folder_closed_loop_tests
anim_name= f'closed_loop_test_KL_mode_{mode}_amp_{amp}.gif'
anim_title= f'KL mode {mode} amp {amp}'

closed_loop_test(num_iterations, gain, leakage, delay, data_kl, anim_path, anim_name, anim_title,
                           RM_S2KL_new, KL2Act_new, Act2KL_new, Phs2KL_new, mask, bias_image, 
                           reference_image=reference_image, diffraction_limited_psf=diffraction_limited_psf,
                           verbose=True, verbose_plot=True)

