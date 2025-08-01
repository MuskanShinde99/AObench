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
import tqdm
from PIL import Image
from matplotlib.colors import LogNorm
from src.utils import *
from collections import deque
from src.dao_setup import init_setup
from src.utils import set_dm_actuators, set_data_dm
from .shm_loader import shm

setup = init_setup()
dm_phase_shm = shm.dm_phase_shm
phase_screen_shm = shm.phase_screen_shm
phase_residuals_shm = shm.phase_residuals_shm
residual_modes_shm = shm.residual_modes_shm
slopes_image_shm = shm.slopes_image_shm
normalized_psf_shm = shm.normalized_psf_shm
computed_modes_shm = shm.computed_modes_shm
commands_shm = shm.commands_shm
dm_kl_modes_shm = shm.dm_kl_modes_shm
dm_act_shm = shm.dm_act_shm


def closed_loop_test(num_iterations, gain, leakage, delay, data_phase_screen, anim_path, aim_name, anim_title,
                           RM_S2KL, KL2Act, Act2KL, Phs2KL,  mask, bias_image, reference_image, diffraction_limited_psf, 
                           verbose=False, verbose_plot=False, **kwargs):
    """
    Performs a closed-loop adaptive optics simulation.
    
    --- Hardware Components ---
    - camera_wfs: Wavefront Sensor (WFS) camera 
    - camera_fp: Focal plane camera for capturing the PSF 
    
    --- Pupil Grid and Mask Parameters ---
    - npix_small_pupil_grid: Number of pixels in the small pupil grid
    - small_pupil_mask: Mask for the small pupil grid field of view
    
    --- Image Processing and Calibration ---
    - mask: Mask for filtering valid pixels
    - bias_image: Bias image used for normalization
    - reference_image: Reference WFS image 
    - diffraction_limited_psf: Ideal PSF used to compute the Strehl ratio
    - RM_S2KL: Matrix mapping PyWFS measurements to KL modes
    - KL2Act: Matrix mapping KL modes to actuator positions
    - Act2KL: Matrix mapping actuator positions back to KL modes
    - Phs2KL: Matrix mapping phase measurements to KL modes (not used in current code, but intended for future use)

    --- General Loop Parameters ---
    - num_iterations: Number of iterations for the AO loop
    - gain: The gain factor for the deformable mirror correction
    - leakage: Leakage term of the AO loop
    - delay: Number of frames of delay in the AO loop
    - data_phase_screen: The phase added to the wavefront
    - anim_path: Path where the output animation is saved
    - aim_name: Name for the saved animation file
    - anim_title: Title of the animation
    """

    # Load hardware and configuration parameters
    camera_wfs = kwargs.get("camera_wfs", setup.camera_wfs)
    camera_fp = kwargs.get("camera_fp", setup.camera_fp)
    npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", setup.npix_small_pupil_grid)
    small_pupil_mask = kwargs.get("small_pupil_mask", setup.pupil_setup.small_pupil_mask)
    
    # Load the folder
    folder_gui = kwargs.get("folder_gui", setup.folder_gui)
    
    # Flatten the DM
    set_data_dm(setup=setup)
    #deformable_mirror.flatten()

    # Reference image 
    normalized_reference_image = normalize_image(reference_image, mask, bias_image)
    pyr_img_shape = reference_image.shape
    if verbose:
        print('Reference image shape:', pyr_img_shape)
    
    # Diffraction limited PSF
    diffraction_limited_psf = diffraction_limited_psf.astype(np.float32)
    diffraction_limited_psf /= diffraction_limited_psf.sum()
    fp_img_shape = diffraction_limited_psf.shape
    if verbose:
        print('PSF shape:', fp_img_shape)
    
    # Create the PSF mask 
    psf_mask, psf_center = create_psf_mask(diffraction_limited_psf, crop_size=100, radius=50)
    # Integrate the flux in that small region
    integrated_diff_psf = diffraction_limited_psf[psf_mask].sum()
    if verbose:
        print('sum center PSF =', integrated_diff_psf)
    
    # Get valid pixel indices from the mask
    valid_pixels_indices = np.where(mask > 0)
    
    # Check delay
    if delay < 0:
        raise ValueError("Delay must be greater than 0.")
        # because delay of negative is not possible in a real AO system
        
    # Initialize queue to store past actuator commands
    max_buffer = delay + 1
    act_pos_queue = deque([np.zeros(setup.nact_total)] * max_buffer, maxlen=max_buffer)
        
    # Initialize variables
    images = []
    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
    
    # Initialize arrays to store Strehl ratio and total residual phase
    strehl_ratios = np.zeros(num_iterations)
    residual_phase_stds = np.zeros(num_iterations)
    
    if verbose_plot:
        # Set up interactive plotting
        plt.ion()
        fig, axs = plt.subplots(4, 2, figsize=(15, 10))
        
        # Initialize display elements
        im_dm = axs[0, 0].imshow(np.zeros_like(data_dm), cmap='viridis')
        axs[0, 0].set_title('DM Phase')
        cbar_dm = plt.colorbar(im_dm, ax=axs[0, 0])
        cbar_dm.set_label('[λ]')
        
        im_phase = axs[0, 1].imshow(np.zeros_like(data_dm), cmap='viridis')
        axs[0, 1].set_title('Phase Screen')
        cbar_phase = plt.colorbar(im_phase, ax=axs[0, 1])
        cbar_phase.set_label('[λ]')
        
        im_pyr = axs[1, 0].imshow(np.zeros(pyr_img_shape), cmap='gray')
        axs[1, 0].set_title('Processed Pyramid Image')
        cbar_pyr = plt.colorbar(im_pyr, ax=axs[1, 0])
        
        im_psf = axs[1, 1].imshow(np.zeros(fp_img_shape), cmap='viridis')
        axs[1, 1].set_title('PSF')
        cbar_psf = plt.colorbar(im_psf, ax=axs[1, 1])
        
        # Initialize additional plots
        line_act, = axs[3, 0].plot(np.zeros(setup.nact_total))
        axs[3, 0].set_title('Commands')
        axs[3, 0].set_xlabel('Actuator Index')
        axs[3, 0].set_ylabel('Amplitude')

        computed_modes = np.zeros(RM_S2KL.shape[1])  # Assuming correct size for KL modes
        line_computed_modes, = axs[3, 1].plot(computed_modes, label='Computed Modes')
        line_residual_modes, = axs[3, 1].plot(computed_modes, label='Residual Modes')
        axs[3, 1].set_title('Computed KL Modes')
        axs[3, 1].set_xlabel('KL Mode Index')
        axs[3, 1].set_ylabel('Amplitude [λ ptp]')
        axs[3, 1].legend()

        line_dm, = axs[2, 0].plot(np.zeros(RM_S2KL.shape[1]))
        axs[2, 0].set_title('Computed Actuator Positions Projected on KL Modes')
        axs[2, 0].set_xlabel('KL Mode')
        axs[2, 0].set_ylabel('Amplitude [λ ptp]')

        im_residuals = axs[2, 1].imshow(np.zeros_like(data_dm), cmap='viridis')
        axs[2, 1].set_title('Phase Residuals')
        cbar_residuals = plt.colorbar(im_residuals, ax=axs[2, 1])
        cbar_residuals.set_label('[λ]')
        
        if verbose:
            print('Dimensions of the phase screen:', data_phase_screen.ndim)

        # Tight layout for better spacing
        plt.tight_layout()

    start_time = time.time()

    for i in tqdm.tqdm(range(num_iterations)):
        
         
        # Determine current phase screen slice
        if data_phase_screen.ndim == 3:
            phase_slice = data_phase_screen[i, :, :]
        elif data_phase_screen.ndim == 2:
            phase_slice = data_phase_screen
        else:
            raise ValueError("data_phase_screen must be either 2D or 3D")
        
        # Update shared memory and image
        phase_screen_shm.set_data(phase_slice) # setting shared memory
        #fits.writeto(os.path.join(folder_gui, f'phase_screen.fits'), phase_slice, overwrite=True)
        
        #Set DM
        current_act_pos = dm_act_shm.get_data().flatten()
        current_act_pos, data_dm, _ = set_data_dm(actuators=current_act_pos, data_phase_screen=phase_slice, setup=setup,)
        dm_phase_shm.set_data(data_dm.astype(np.float32)* small_pupil_mask)  # setting shared memory
     
        # Compute phase residuals 
        phase_residuals = (phase_slice + data_dm) * small_pupil_mask
        phase_residuals_shm.set_data(phase_residuals) # setting shared memory
        #fits.writeto(os.path.join(folder_gui, f'phase_residuals.fits'), phase_residuals, overwrite=True)

        residual_phase_std = np.std(phase_residuals[small_pupil_mask == 1])
        if data_phase_screen.ndim == 3:
            residual_phase_stds[i] = residual_phase_std
                
        # Compute residual modes
        residual_modes = phase_residuals.flatten() @ Phs2KL
        residual_modes_shm.set_data(residual_modes) # setting shared memory


        # Capture and process WFS image
        slopes_image = get_slopes_image(
            mask,
            bias_image,
            normalized_reference_image,
            setup=setup,
        )
        slopes_image_shm.set_data(slopes_image)
        slopes = slopes_image[valid_pixels_indices].flatten()
        #fits.writeto(os.path.join(folder_gui, f'slopes_image.fits'), slopes_image, overwrite=True)

        # Capture PSF
        fp_img = camera_fp.get_data()
        fp_img = np.maximum(fp_img, 1e-10)
        normalized_psf_shm.set_data((fp_img / np.max(fp_img))) # setting shared memory
        #fits.writeto(os.path.join(folder_gui, f'normalized_psf.fits'), (fp_img / np.max(fp_img)), overwrite=True)
        
        # Compute Strehl ratio
        observed_psf = fp_img / fp_img.sum()
        integrated_obs_psf = observed_psf[psf_mask].sum()
        strehl_ratio = integrated_obs_psf / integrated_diff_psf
        # strehl_ratio = np.max(observed_psf) / np.max(diffraction_limited_psf)
        strehl_ratios[i] = strehl_ratio

        # Compute KL modes present
        computed_modes = slopes @ RM_S2KL 
        # multiply by two because this mode is computed for DM surface and we want DM phase
        computed_modes_shm.set_data(computed_modes) # setting shared memory
        
        # Compute actuator commands
        act_pos = computed_modes @ KL2Act
        commands_shm.set_data(act_pos) # setting shared memory
        
        # Append current actuator command
        act_pos_queue.append(act_pos)
        
        # Retrieve delayed actuator command (0 = current, 1 = 1-step delay, etc.)
        delayed_act_pos = act_pos_queue[-(max_buffer)]  
        
        # Apply delayed actuator command
        # deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * delayed_act_pos
        new_act_pos = (1 - leakage) * current_act_pos - gain * delayed_act_pos
        set_dm_actuators(new_act_pos, setup=setup,)
        
        # KL modes corresponding the total actuator postion or the DM shape
        modes_act_pos_tot = current_act_pos @ Act2KL
        dm_kl_modes_shm.set_data(modes_act_pos_tot) # setting shared memory


        if verbose_plot:
            # Update deformable mirror phase map
            im_dm.set_data(data_dm)
            im_dm.set_clim(np.min(data_dm), np.max(data_dm))
            cbar_dm.update_normal(im_dm)

            # Update current phase screen 
            im_phase.set_data(phase_slice)
            im_phase.set_clim(np.min(phase_slice), np.max(phase_slice))
            cbar_phase.update_normal(im_phase)

            # Update residual phase map (after DM correction)
            im_residuals.set_data(phase_residuals)
            im_residuals.set_clim(np.min(phase_residuals), np.max(phase_residuals))
            cbar_residuals.update_normal(im_residuals)
            axs[2, 1].set_title(f'Total residuals std: {residual_phase_std:.4f} x λ') # the phase residuals are in  waves

            # Update processed WFS image (slopes map from PWFS)
            im_pyr.set_data(slopes_image)
            im_pyr.set_clim(np.min(slopes_image), np.max(slopes_image))
            cbar_pyr.update_normal(im_pyr)

            # Update normalized PSF (focal-plane image)
            im_psf.set_data((fp_img / np.max(fp_img)))
            im_psf.set_norm(LogNorm(vmin=1e-6, vmax=1e1))
            cbar_psf.update_normal(im_psf)
            axs[1, 1].set_title(f'PSF Strehl {strehl_ratio:.4f}')

            # Update actuator commands (raw actuator vector)
            line_act.set_ydata(act_pos)
            axs[3, 0].set_ylim(np.min(act_pos), np.max(act_pos))

            # Update KL mode plots (computed and residual)
            line_residual_modes.set_ydata(residual_modes)
            line_computed_modes.set_ydata(computed_modes)
            axs[3, 1].set_ylim(np.min(residual_modes), np.max(residual_modes))

            # Update KL projection of DM shape
            line_dm.set_ydata(modes_act_pos_tot)
            axs[2, 0].set_ylim(np.min(modes_act_pos_tot), np.max(modes_act_pos_tot))

            # Finalize figure
            fig.subplots_adjust(top=0.9)
            fig.suptitle(f'AO Bench -- {anim_title} - Iteration {i+1}')
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_data = img_data[:, :, :3]
            images.append(Image.fromarray(img_data))
            plt.draw()
            plt.pause(0.1)


    if verbose_plot:
        plt.ioff()
        plt.show()
        images[0].save(os.path.join(anim_path, aim_name), save_all=True, append_images=images[1:], duration=300, loop=0)
    
    end_time = time.time()
    if verbose:
        print(f'Total time for AO loop: {end_time - start_time:.2f} s')
    
    return strehl_ratios, residual_phase_stds
