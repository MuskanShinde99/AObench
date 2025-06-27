#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 09:36:03 2025

@author: ristretto-dao
"""

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
