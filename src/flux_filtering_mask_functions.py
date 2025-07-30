#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:19:48 2025

@author: laboptic
"""

import numpy as np
import time
import os
from matplotlib import pyplot as plt
from src.dao_setup import init_setup
from src.tilt_functions import apply_intensity_tilt_kl
from src.utils import set_data_dm
from astropy.io import fits

setup = init_setup()



def create_summed_image_for_mask(modulation_angles, modulation_amp, tiltx, tilty, verbose=False, **kwargs):
    """Acquire and sum images for flux mask generation using tip-tilt modulation.

    Parameters
    ----------
    modulation_angles : array-like
        Tip/tilt angles (degrees) to apply on the deformable mirror.
    modulation_amp : float
        Amplitude of the modulation in actuator units.
    tiltx : ndarray
        X component of the tip/tilt mode.
    tilty : ndarray
        Y component of the tip/tilt mode.
    verbose : bool, optional
        If ``True`` print progress information.
    camera : optional
        Camera object used for acquisition (default ``setup.camera_wfs``).

    Returns
    -------
    numpy.ndarray
        Summed image resulting from the modulation sequence.
    """
    
    # Use kwargs or default from setup
    camera = kwargs.get("camera", setup.camera_wfs)

    modulation_img_arr = []

    for angle in modulation_angles:
        if verbose:
            print(f"Applying modulation angle: {angle}")
        
        #modulation_step = apply_intensity_tilt(small_pupil_mask, Npix / 2, tilt_angle=angle)
        modulation_step = apply_intensity_tilt_kl(tiltx, tilty, tilt_angle=angle)
        data_modulation = modulation_amp * modulation_step
        
        # Put on DM
        actuators, _, _  = set_data_dm(data_modulation, setup=setup,)

        # Capture image using pylon SDK wrapper function
        n_frames=50
        img = (np.mean([camera.get_data() for i in range(n_frames)], axis=0))
        #img = camera.get_data(check=True)
        
        modulation_img_arr.append(img)

    summed_image = np.sum(np.asarray(modulation_img_arr), axis=0)

    return summed_image


def create_summed_image_for_mask_dm_random(n_iter, verbose=False, **kwargs):
    """Acquire a summed image using random DM actuator patterns.

    Parameters
    ----------
    n_iter : int
        Number of random actuator patterns to apply.
    verbose : bool, optional
        If ``True`` display progress information.
    camera : optional
        Camera object used for acquisition (default ``setup.camera_wfs``).
    nact_total : int, optional
        Number of DM actuators (default ``setup.nact_total``).

    Returns
    -------
    numpy.ndarray
        Summed image produced by random push--pull modulation.
    """

    # Use kwargs or defaults from the setup
    camera = kwargs.get("camera", setup.camera_wfs)
    nact_total = kwargs.get("nact_total", setup.nact_total)
    nact_valid = kwargs.get("nact_valid", setup.nact_valid)

    img_arr = []


    for i in range(n_iter):
        if verbose:
            print(f"Iteration {i + 1}")

        #act_random = np.random.choice([0, 0.1], size=nact_total)
        act_random = np.random.choice([-1, 1], size=nact_valid)
        actuators, _, _  = set_data_dm(act_random, setup=setup,)
        
        n_frames=20
        img = (np.mean([camera.get_data() for i in range(n_frames)], axis=0))
        img_arr.append(img)

    summed_image = np.sum(np.asarray(img_arr), axis=0)

    return summed_image

def create_flux_filtering_mask(method, flux_cutoff, tiltx, tilty,
                               modulation_angles=np.arange(0, 360, 10), modulation_amp=15, n_iter=200,
                               create_summed_image=True, verbose=False, verbose_plot=False, **kwargs):
    """Generate a binary mask highlighting high-flux regions.

    Parameters
    ----------
    method : str
        Acquisition strategy, ``'tip_tilt_modulation'`` or ``'dm_random'``.
    flux_cutoff : float
        Relative intensity threshold used to build the mask.
    tiltx, tilty : ndarray
        Tip/tilt modes used when ``method='tip_tilt_modulation'``.
    modulation_angles : array-like, optional
        Modulation angles in degrees for tip-tilt modulation.
    modulation_amp : float, optional
        Modulation amplitude for tip-tilt modulation.
    n_iter : int, optional
        Number of iterations for the ``'dm_random'`` method.
    create_summed_image : bool, optional
        If ``True`` compute a new summed image, otherwise load the previous one.
    verbose : bool, optional
        Print progress information.
    verbose_plot : bool, optional
        Display intermediate plots.
    **kwargs : optional
        ``camera`` and ``nact_total`` passed to image acquisition functions,
        ``folder_pyr_mask`` and ``folder_calib`` to override output locations.

    Returns
    -------
    numpy.ndarray
        Binary mask highlighting high-flux regions.
    """

    folder_pyr_mask = kwargs.get("folder_pyr_mask", setup.folder_pyr_mask)
    folder_calib = kwargs.get("folder_calib", setup.folder_calib)

    summed_img_path = os.path.join(folder_calib, f'binned_summed_pyr_images_3s_pyr.fits')

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
                verbose=verbose,
                **kwargs
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
        os.path.join(folder_calib, f'binned_masked_pyr_images_3s_pyr.fits'),
        masked_summed_image.astype(np.float32), overwrite=True
    )
    fits.writeto(
        os.path.join(folder_calib, f'binned_mask_3s_pyr.fits'),
        mask.astype(np.uint8), overwrite=True
    )

    return mask
