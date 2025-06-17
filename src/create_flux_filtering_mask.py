# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:49:22 2024

@author: RISTRETTO
"""

from src.tilt import apply_intensity_tilt
import numpy as np
from pypylon import pylon
from DEVICES_3.Basler_Pylon.test_pylon import pylonGrab
from hcipy import *
import time
from matplotlib import pyplot as plt

def create_flux_filtering_mask(modulation_angles, modulation_amp, flux_cutoff, camera, slm, pupil_data, pupil_size, verbose=False):
    """
    Create a mask based on the modulation angles and flux cutoff.

    Parameters:
    - modulation_angles: list of angles for modulation
    - modulation_amp??
    - flux_cutoff: cutoff factor for flux limit
    - camera: initialized camera object
    - slm: SLM instance
    - pupil_data: data of the pupil
    - pupil_size: size of the pupil in mm
    - verbose: if True, display each captured image

    Returns:
    - mask: generated mask as a boolean numpy array
    - summed_image: the summed image used for mask creation
    """
    dataWidth = slm.width_px
    dataHeight = slm.height_px
    pixel_size_mm = slm.pixelsize_m * 1e3  # pixel size in mm
    Npix = pupil_size / pixel_size_mm  # replaced npix_pupil with Npix

    # Initialize arrays for captured images
    modulation_img_arr = []

    # Prepare a pupil grid
    pupil_grid = make_pupil_grid([dataWidth, dataHeight], [dataWidth * pixel_size_mm, dataHeight * pixel_size_mm])
    vlt_aperture_generator = make_obstructed_circular_aperture(pupil_size, 0, 0, 0)
    vlt_aperture = evaluate_supersampled(vlt_aperture_generator, pupil_grid, 1)

    for angle in modulation_angles:
        # Apply modulation step
        modulation_step = apply_intensity_tilt(vlt_aperture.shaped, Npix / 2, tilt_angle=angle)  # Updated to use Npix
        modulation_data = (pupil_data + modulation_amp * modulation_step) % 1  # Update and wrap data

        # Show data on the SLM
        error = slm.showData(modulation_data)
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
        
        # Wait for the SLM to update
        time.sleep(0.3)

        # Capture image
        img = pylonGrab(camera, 1)
        height, width = img.shape[:2]
        img_size = min(height, width)
        img = img[0:img_size, 0:img_size]
        modulation_img_arr.append(img)

        # Display the captured image if verbose is True
        if verbose:
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.colorbar()
            plt.title(f'Captured Image at {angle} Degrees; max {np.max(img)}')
            plt.show()

    # Compute the summed image
    summed_image = np.sum(np.asarray(modulation_img_arr), axis=0)

    # # Create a filtering mask
    # mask = np.zeros(summed_image.shape, dtype=bool)
    # flux_limit_upper = summed_image.max() * flux_cutoff
    # mask[summed_image >= flux_limit_upper] = True

    return summed_image
