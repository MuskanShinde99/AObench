#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:29 2025

@author: laboptic
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import LogNorm
from src.utils import *
from hcipy import *
from DEVICES_3.Basler_Pylon.test_pylon import *
from collections import deque

def closed_loop_test(num_iterations, gain, leakage, delay, data_phase_screen, anim_path, aim_name, anim_title,
                           RM_PyWFS2KL, KL2Act, Act2KL, Phs2KL,  mask, bias_image, **kwargs):
    """
    Performs a closed-loop adaptive optics simulation.
    
    --- Hardware Components ---
    - deformable_mirror: The deformable mirror for wavefront correction
    - slm: Spatial Light Modulator (SLM) for displaying the correction
    - camera_wfs: Wavefront Sensor (WFS) camera used for wavefront analysis
    - camera_fp: Focal plane camera for capturing the PSF (Point Spread Function)
    
    --- Pupil Grid and Mask Parameters ---
    - npix_small_pupil_grid: Number of pixels in the small pupil grid
    - data_pupil: Data to create pupil on slm
    - data_pupil_outer: Outer region of the pupil data
    - data_pupil_inner: Inner region of the pupil data
    - pupil_mask: Mask for the pupil region
    - small_pupil_mask: Mask for the small pupil grid field of view
    
    --- Image Processing and Calibration ---
    - mask: Mask for filtering unwanted parts of the image (e.g., noise)
    - bias_image: Bias image used for normalization
    - RM_PyWFS2KL: Matrix mapping PyWFS measurements to KL modes
    - KL2Act: Matrix mapping KL modes to actuator positions
    - Act2KL: Matrix mapping actuator positions back to KL modes
    - Phs2KL: Matrix mapping phase measurements to KL modes (not used in current code, but intended for future use)

    --- General Loop Parameters ---
    - num_iterations: Number of iterations for the simulation loop
    - gain: The gain factor for the deformable mirror correction
    - leakage: Leakage term of the AO loop
    - delay: Number of frames of delay in the AO loop
    - data_phase_screen: The phase added to the wavefront
    - anim_path: Path where the output animation is saved
    - aim_name: Name for the saved animation file
    - anim_title: Title of the animation
    """
    
    # Load hardware and configuration parameters
    deformable_mirror = kwargs.get("deformable_mirror", deformable_mirror)
    slm = kwargs.get("slm", slm)
    camera_wfs = kwargs.get("camera_wfs", camera_wfs)
    camera_fp = kwargs.get("camera_fp", camera_fp)
    npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", npix_small_pupil_grid)
    data_pupil = kwargs.get("data_pupil", data_pupil)
    data_pupil_outer = kwargs.get("data_pupil_outer", data_pupil_outer)
    data_pupil_inner = kwargs.get("data_pupil_inner", data_pupil_inner)
    pupil_mask = kwargs.get("pupil_mask", pupil_mask)
    small_pupil_mask = kwargs.get("small_pupil_mask", small_pupil_mask)
    
    # Display Pupil Data on SLM
    data_slm = compute_data_slm()
    slm.set_data(data_slm)
        
    # Capture a reference image using the WFS camera
    time.sleep(wait_time)  # Wait for stabilization of SLM
    reference_image = camera_wfs.get_data()
    normalized_reference_image = normalize_image(reference_image, mask, bias_image)
    pyr_img_shape = reference_image.shape
    print('Reference image shape:', pyr_img_shape)
    
    # Diffraction limited PSF
    diffraction_limited_psf = camera_fp.get_data().astype(np.float32)
    diffraction_limited_psf /= diffraction_limited_psf.sum()
    fp_img_shape = diffraction_limited_psf.shape
    print('PSF shape:', fp_img_shape)
    
    # Create the PSF mask 
    psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)
    # Integrate the flux in that small region
    integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()
    print('sum center PSF =', integrated_diff_psf)
    
    # Get valid pixel indices from the mask
    valid_pixels_indices = np.where(mask > 0)
    
    #Check delay
    if delay < 0:
        raise ValueError("Delay must be greater than 0.")
        # because delay of negative is not possible in a real AO system
        
    # Initialize queue to store past actuator commands
    max_buffer = delay + 1
    act_pos_queue = deque([np.zeros(deformable_mirror.num_actuators)] * max_buffer, maxlen=max_buffer)
        
    # Initialize variables
    images = []
    deformable_mirror.flatten()
    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
    
    # Initialize arrays to store Strehl ratio and total residual phase
    strehl_ratios = np.zeros(num_iterations)
    residual_phase_stds = np.zeros(num_iterations)
    
    
    # Set up interactive plotting
    plt.ion()
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))
    
    # Initialize display elements
    im_dm = axs[0, 0].imshow(np.zeros_like(data_dm), cmap='viridis')
    axs[0, 0].set_title('DM Phase')
    cbar_dm = plt.colorbar(im_dm, ax=axs[0, 0])
    cbar_dm.set_label('[2π rad]')
    
    im_phase = axs[0, 1].imshow(np.zeros_like(data_dm), cmap='viridis')
    axs[0, 1].set_title('Phase Screen')
    cbar_phase = plt.colorbar(im_phase, ax=axs[0, 1])
    cbar_phase.set_label('[2π rad]')
    
    im_pyr = axs[1, 0].imshow(np.zeros(pyr_img_shape), cmap='gray')
    axs[1, 0].set_title('Processed Pyramid Image')
    cbar_pyr = plt.colorbar(im_pyr, ax=axs[1, 0])
    
    im_psf = axs[1, 1].imshow(np.zeros(fp_img_shape), cmap='viridis')
    axs[1, 1].set_title('PSF')
    cbar_psf = plt.colorbar(im_psf, ax=axs[1, 1])
    
    # Initialize additional plots
    line_act, = axs[3, 0].plot(np.zeros(deformable_mirror.num_actuators))
    axs[3, 0].set_title('Commands')
    axs[3, 0].set_xlabel('Actuator Index')
    axs[3, 0].set_ylabel('Amplitude')

    computed_modes = np.zeros(RM_PyWFS2KL.shape[1])  # Assuming correct size for KL modes
    line_computed_modes, = axs[3, 1].plot(computed_modes, label='Computed Modes')
    line_residual_modes, = axs[3, 1].plot(computed_modes, label='Residual Modes')
    axs[3, 1].set_title('Computed KL Modes')
    axs[3, 1].set_xlabel('KL Mode Index')
    axs[3, 1].set_ylabel('Amplitude [2π rad ptp]')
    axs[3, 1].legend()

    line_dm, = axs[2, 0].plot(np.zeros(RM_PyWFS2KL.shape[1]))
    axs[2, 0].set_title('Computed Actuator Positions Projected on KL Modes')
    axs[2, 0].set_xlabel('KL Mode')
    axs[2, 0].set_ylabel('Amplitude [2π rad ptp]')

    im_residuals = axs[2, 1].imshow(np.zeros_like(data_dm), cmap='viridis')
    axs[2, 1].set_title('Phase Residuals')
    cbar_residuals = plt.colorbar(im_residuals, ax=axs[2, 1])
    cbar_residuals.set_label('[2π rad]')
    
    print('Dimensions of the phase screen:', data_phase_screen.ndim)

    # Tight layout for better spacing
    plt.tight_layout()

    start_time = time.time()
    
    from tqdm import tqdm

    for i in tqdm(range(num_iterations)):
        # Update deformable mirror surface
        data_dm[:, :] = deformable_mirror.opd.shaped
        im_dm.set_data(data_dm)
        im_dm.set_clim(np.min(data_dm), np.max(data_dm))
        cbar_dm.update_normal(im_dm)
        
        # Handle 2D or 3D `data_phase_screen`
        if data_phase_screen.ndim == 3:
            
            # Update phase screen
            # If 3D, use the i-th slice (frame)
            im_phase.set_data(data_phase_screen[i, :, :])
            im_phase.set_clim(np.min(data_phase_screen[i, :, :]), np.max(data_phase_screen[i, :, :]))
            cbar_phase.update_normal(im_phase)
            
            # Residual phase
            phase_residuals = (data_phase_screen[i, :, :] + data_dm)*small_pupil_mask
            residual_modes = phase_residuals.flatten() @ Phs2KL
            im_residuals.set_data(phase_residuals)
            im_residuals.set_clim(np.min(phase_residuals), np.max(phase_residuals))
            cbar_residuals.update_normal(im_residuals)
            residual_phase_std = np.std(phase_residuals[small_pupil_mask==1])
            residual_phase_stds[i] = residual_phase_std
            axs[2, 1].set_title(f'Total residuals std: {residual_phase_std:.4f} x 2π rad') # the phase residuals are in  waves. To display in rad, multiply by 2*np.pi
            
            # Compute SLM data
            data_slm = compute_data_slm(data_dm=data_dm, data_phase_screen=data_phase_screen[i, :, :])

        elif data_phase_screen.ndim == 2:
            
            # Update phase screen
            # If 2D, use the entire data_phase_screen as is
            im_phase.set_data(data_phase_screen)
            im_phase.set_clim(np.min(data_phase_screen), np.max(data_phase_screen))
            cbar_phase.update_normal(im_phase)
            
            # Residual phase
            phase_residuals = (data_phase_screen + data_dm)*small_pupil_mask
            residual_modes = phase_residuals.flatten() @ Phs2KL
            im_residuals.set_data(phase_residuals)
            im_residuals.set_clim(np.min(phase_residuals), np.max(phase_residuals))
            cbar_residuals.update_normal(im_residuals)
            axs[2, 1].set_title(f'Total residuals std: {np.std(phase_residuals[phase_residuals != 0]):.4f} x 2π rad') # the phase residuals are in  waves. To display in rad, multiply by 2*np.pi
            
            # Compute SLM data
            data_slm = compute_data_slm(data_dm=data_dm, data_phase_screen=data_phase_screen)

            
        else:
            raise ValueError("data_phase_screen must be either 2D or 3D")
            
        # Send data to SLM
        slm.set_data(data_slm)

        # Allow SLM to settle and capture a Pyramid image
        time.sleep(wait_time)
        pyr_img = camera_wfs.get_data()
        
        # Process the Pyramid image
        normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
        slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)
        slopes = slopes_image[valid_pixels_indices].flatten() #* 2 #trial factor

        im_pyr.set_data(slopes_image)
        im_pyr.set_clim(np.min(slopes_image), np.max(slopes_image))
        #im_pyr.set_clim(0.002, -0.001)
        cbar_pyr.update_normal(im_pyr)

        # Capture and process the PSF
        fp_img = camera_fp.get_data()
        fp_img = np.maximum(fp_img, 1e-10)
        im_psf.set_data((fp_img/np.max(fp_img)))
        im_psf.set_norm(LogNorm(vmin=1e-6, vmax=1e1))
        cbar_psf.update_normal(im_psf)

        # # Compute Strehl ratio
        # observed_psf = fp_img / fp_img.sum()
        # integrated_obs_psf = observed_psf[psf_mask].sum()
        # strehl_ratio = integrated_obs_psf / integrated_diff_psf
        # axs[1, 1].set_title(f'PSF Strehl {strehl_ratio:.4f}')
        
        # Compute Strehl ratio
        observed_psf = fp_img / fp_img.sum()
        strehl_ratio = np.max(observed_psf) / np.max(diffraction_limited_psf)
        strehl_ratios[i] = strehl_ratio
        axs[1, 1].set_title(f'PSF Strehl {strehl_ratio:.4f}')

        # Compute actuator commands
        computed_modes = slopes @ RM_PyWFS2KL
        act_pos = computed_modes @ KL2Act
        
        # Append current actuator command
        act_pos_queue.append(act_pos)
        
        # Retrieve delayed actuator command (0 = current, 1 = 1-step delay, etc.)
        delayed_act_pos = act_pos_queue[-(max_buffer)]  

        # Apply delayed actuator command
        deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * delayed_act_pos
        #deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * act_pos
        
        act_pos_tot = deformable_mirror.actuators @ Act2KL

        # Update actuator and mode plots
        line_act.set_ydata(act_pos)
        axs[3, 0].set_ylim(np.min(act_pos), np.max(act_pos))
        line_residual_modes.set_ydata(residual_modes)
        line_computed_modes.set_ydata(computed_modes)
        axs[3, 1].set_ylim(np.min(residual_modes), np.max(residual_modes))
        #axs[3, 1].set_title(f'Total residuals: {np.sqrt(np.sum(residual_modes**2)):.4f}')
        line_dm.set_ydata(act_pos_tot)
        axs[2, 0].set_ylim(np.min(act_pos_tot), np.max(act_pos_tot))
        
        fig.subplots_adjust(top=0.9)  # Moves everything slightly down
        fig.suptitle(f'AO Bench -- {anim_title} - Iteration {i+1}')
        
        # Capture the plot for the GIF
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_data = img_data[:, :, :3]  # Remove alpha channel (RGBA -> RGB)
        images.append(Image.fromarray(img_data))

        # Redraw and pause for real-time display
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    
    # Save animation
    images[0].save(os.path.join(anim_path, aim_name), save_all=True, append_images=images[1:], duration=300, loop=0)
    
    end_time = time.time()
    print(f'Total time for AO loop: {end_time - start_time:.2f} s')
    
    return strehl_ratios, residual_phase_stds

