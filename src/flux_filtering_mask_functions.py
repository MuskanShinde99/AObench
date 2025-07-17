#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:19:48 2025

@author: laboptic
"""

import numpy as np
from pypylon import pylon
from DEVICES_3.Basler_Pylon.test_pylon import *
from hcipy import *
import time
from matplotlib import pyplot as plt
from src.dao_setup import *  # Import all variables from setup
import src.dao_setup as dao_setup
from src.tilt_functions import *
from src.utils import *
import matplotlib.colors as mcolors



def create_summed_image_for_mask(modulation_angles, modulation_amp, tiltx, tilty, verbose=False, **kwargs):
    """
    Create a mask based on the modulation angles and flux cutoff.

    Parameters:
    - modulation_angles: list of angles for modulation
    - modulation_amp: modulation amplitude
    - verbose: if True, print info messages
    - verbose_plot: if True, update the displayed image dynamically
    - kwargs: optional overrides for:
        - camera: camera object 
        - slm: spatial light modulator object 
        - data_pupil: pupil data array 
        - pupil_size: pupil size in mm 
        - pupil_mask (ndarray): Boolean mask for pupil over the full slm.
        - small_pupil_mask (ndarray): Boolean mask for small square pupil on the slm.

    Returns:
    - summed_image: the summed image used for mask creation
    """
    
    from src.dao_setup import wait_time

    # Use kwargs or default from dao_setup
    camera = kwargs.get("camera", dao_setup.camera_wfs)
    slm = kwargs.get("slm", dao_setup.slm)
    deformable_mirror = kwargs.get("slm", deformable_mirror.slm)
    data_pupil = kwargs.get("data_pupil", dao_setup.data_pupil)
    pupil_size = kwargs.get("pupil_size", dao_setup.pupil_size)
    pixel_size = kwargs.get("pixel_size", dao_setup.pixel_size)
    pupil_mask = kwargs.get("pupil_mask", dao_setup.pupil_mask)
    small_pupil_mask = kwargs.get("small_pupil_mask", dao_setup.small_pupil_mask)
    dataHeight = kwargs.get("dataHeight", dao_setup.dataHeight)
    dataWidth = kwargs.get("dataWidth", dao_setup.dataWidth)
    npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", dao_setup.npix_small_pupil_grid)

    pixel_size_mm = pixel_size
    Npix = pupil_size / pixel_size_mm  # replaced npix_pupil with Npix

    modulation_img_arr = []

    for angle in modulation_angles:
        if verbose:
            print(f"Applying modulation angle: {angle}")
        
        #modulation_step = apply_intensity_tilt(small_pupil_mask, Npix / 2, tilt_angle=angle)
        modulation_step = apply_intensity_tilt_kl(tiltx, tilty, small_pupil_mask, tilt_angle=angle)
        data_modulation = modulation_amp * modulation_step
        
        # Put on DM
        data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
        deformable_mirror.flatten()
        deformable_mirror.actuators = data_modulation  
        data_dm[:, :] = deformable_mirror.opd.shaped/2 
        data_slm = compute_data_slm(data_dm=data_dm)
        slm.set_data(data_slm)

        time.sleep(wait_time)  # wait for SLM update

        # Capture image using pylon SDK wrapper function
        img = camera.get_data()
        modulation_img_arr.append(img)

    summed_image = np.sum(np.asarray(modulation_img_arr), axis=0)

    return summed_image


def create_summed_image_for_mask_dm_random(n_iter, verbose=False, **kwargs):
    """
    Run push-pull image acquisition using random DM actuator patterns.

    Parameters:
    - n_iter: number of random actuator patterns to test
    - verbose: whether to print/display progress
    - verbose_plot: if True, live image display is updated
    - kwargs:
        - slm: SLM instance (default: slm)
        - camera: initialized camera object (default: camera_wfs)
        - deformable_mirror: DM object with actuator control (default: deformable_mirror)
        - npix_small_pupil_grid: resolution of the inner grid (default: npix_small_pupil_grid)

    Returns:
    - summed_image: result of push-pull image subtraction summed over all modes
    """

    slm = kwargs.get("slm", dao_setup.slm)
    camera = kwargs.get("camera", dao_setup.camera_wfs)
    deformable_mirror = kwargs.get("deformable_mirror", dao_setup.deformable_mirror)
    deformable_mirror = DeformableMirror(dm_modes_full)
    npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", dao_setup.npix_small_pupil_grid)

    nact_total = int(deformable_mirror.num_actuators)
    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)

    img_arr = []


    for i in range(n_iter):
        if verbose:
            print(f"Iteration {i + 1}")

        act_random = np.random.choice([0, 1], size=nact_total)
        deformable_mirror.actuators = act_random
        data_dm[:, :] = deformable_mirror.opd.shaped/2
        
        data_slm = compute_data_slm(data_dm=data_dm)
        slm.set_data(data_slm)
        time.sleep(wait_time)
        
        img = camera.get_data()
        img_arr.append(img)

    summed_image = np.sum(np.asarray(img_arr), axis=0)

    return summed_image

def create_flux_filtering_mask(method, flux_cutoff, tiltx, tilty,
                               modulation_angles=np.arange(0, 360, 10), modulation_amp=15, n_iter=200,
                               create_summed_image=True, verbose=False, verbose_plot=False, **kwargs):
    """
    Create a flux filtering mask.

    Parameters:
    - method (str): Type of summed image acquisition. Options:
        * 'tip_tilt_modulation' → uses `create_summed_image_for_mask`
        * 'dm_random' → uses `create_summed_image_for_mask_dm_random`
    - flux_cutoff (float): Relative intensity threshold to generate binary mask (e.g., 0.61)
    - modulation_angles (array): Tip-tilt modulation angles in degrees (only used in 'tip_tilt_modulation')
    - modulation_amp (float): Tip-tilt modulation amplitude (only used in 'tip_tilt_modulation')
    - n_iter (int): Number of iterations (only used in 'dm_random')
    - create_summed_image (bool): If True, compute new summed image; otherwise load from disk
    - verbose (bool): Print progress info
    - verbose_plot (bool): Show real-time image display
    - **kwargs: Additional parameters passed to the internal image generation functions

    Returns:
    - mask (np.ndarray): Binary mask highlighting high-flux regions
    """

    folder_pyr_mask = kwargs.get("folder_pyr_mask", dao_setup.folder_pyr_mask)
    folder_calib = kwargs.get("folder_calib", dao_setup.folder_calib)
    pupil_size = kwargs.get("pupil_size", dao_setup.pupil_size)

    summed_img_path = os.path.join(folder_calib, f'binned_summed_pyr_images_pup_{pupil_size}mm_3s_pyr.fits')

    if create_summed_image:
        if verbose:
            print(f'Creating summed image using method: {method}')

        if method == 'tip_tilt_modulation':
            summed_image = create_summed_image_for_mask(
                modulation_angles, modulation_amp, tiltx, tilty,
                verbose=verbose, **kwargs
            )
        elif method == 'dm_random':
            summed_image = create_summed_image_for_mask_dm_random(
                n_iter=n_iter,
                verbose=verbose, **kwargs
            )
        else:
            raise ValueError("Invalid method. Use 'tip_tilt_modulation' or 'dm_random'.")

        fits.writeto(summed_img_path, summed_image.astype(np.float32), overwrite=True)
        if verbose:
            print(f'Summed image saved to: {summed_img_path}')
    else:
        if verbose:
            print(f'Loading summed image from: {summed_img_path}')
        summed_image = fits.getdata(summed_img_path)

    if verbose_plot:
        plt.figure()
        plt.title('Summed Image')
        plt.imshow(summed_image, cmap='viridis')
        plt.colorbar()
        # plt.savefig(os.path.join(folder_pyr_mask, f'summed_image_dm_modulation.png'))
        plt.show()

    flux_limit_upper = summed_image.max() * flux_cutoff
    mask = summed_image >= flux_limit_upper

    if verbose:
        print(f'Flux filtering mask created using cutoff {flux_cutoff:.2f}')

    if verbose_plot:
        plt.figure()
        plt.title('Flux Filtering Mask')
        plt.imshow(mask, cmap='viridis')
        plt.colorbar()
        # plt.savefig(os.path.join(folder_pyr_mask, f'mask_dm_modulation.png'))
        plt.show()

    masked_summed_image = mask * summed_image

    if verbose_plot:
        plt.figure()
        plt.title('Masked Summed Image')
        plt.imshow(masked_summed_image, cmap='viridis')
        plt.colorbar()
        # plt.savefig(os.path.join(folder_pyr_mask, f'masked_summed_image_dm_modulation.png'))
        plt.show()

    if verbose:
        print('Saving masked image and mask')

    fits.writeto(
        os.path.join(folder_calib, f'binned_masked_pyr_images_pup_{pupil_size}mm_3s_pyr.fits'),
        masked_summed_image.astype(np.float32), overwrite=True
    )
    fits.writeto(
        os.path.join(folder_calib, f'binned_mask_pup_{pupil_size}mm_3s_pyr.fits'),
        mask.astype(np.uint8), overwrite=True
    )

    return mask