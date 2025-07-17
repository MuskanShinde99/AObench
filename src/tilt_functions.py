# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:40:08 2024

@author: RISTRETTO
"""

# tilt.py

import numpy as np
from matplotlib import pyplot as plt



def apply_intensity_tilt(image, pupil_radius, tilt_angle):
    """
    Apply an intensity tilt effect to the image within the pupil region at a specific angle,
    with the tilt effect ranging from -1 to 1 within the pupil.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - pupil_radius: The radius of the pupil in pixels.
    - tilt_angle: The angle of the intensity tilt in degrees.
    
    Returns:
    - The image with the intensity tilt applied within the pupil region.
    """
    dataHeight, dataWidth = image.shape
    x = np.linspace(-dataWidth / 2, dataWidth / 2, dataWidth)
    y = np.linspace(-dataHeight / 2, dataHeight / 2, dataHeight)
    xx, yy = np.meshgrid(x, y)
    
    # Define pupil mask within the tilt function
    rr = np.abs(xx + 1j * yy)
    mask = rr < pupil_radius
    
    # Convert angle to radians
    tilt_angle_rad = np.deg2rad(tilt_angle)
    
    # Compute the tilt effect
    x_normalized = xx / pupil_radius  # Normalize by pupil radius
    y_normalized = yy / pupil_radius  # Normalize by pupil radius
    
    # Tilt effect based on angle
    tilt_effect = (x_normalized * np.cos(tilt_angle_rad) + 
                   y_normalized * np.sin(tilt_angle_rad))
    
    
    # Scale the tilt effect to [-1, 1] within the mask
    tilt_min = tilt_effect[mask].min()
    tilt_max = tilt_effect[mask].max()
    tilt_effect[mask] = 2 * (tilt_effect[mask] - tilt_min) / (tilt_max - tilt_min) - 1

    # Mask outside the pupil
    tilt_effect[~mask] = 0
    
    return tilt_effect/np.ptp(tilt_effect)



def apply_intensity_tilt_kl(tiltx, tilty, mask, tilt_angle):
    """
    Apply an intensity tilt effect at a specific angle,
    with the tilt effect ranging from -1 to 1 within the pupil.
    
    Parameters:
    - tiltx: The tilt in x direction.
    - tilty: The tilt in y direction.
    - tilt_angle: The angle of the intensity tilt in degrees.
    
    Returns:
    - The data with the intensity tilt applied.
    """
    
    # Convert angle to radians
    tilt_angle_rad = np.deg2rad(tilt_angle)

    # Tilt effect based on angle
    tilt_effect = (tiltx * np.cos(tilt_angle_rad) + 
                   tilty * np.sin(tilt_angle_rad))
    
    return tilt_effect

def apply_intensity_tilt_DM(tiltx, tilty, tilt_angle):
    """
    Apply an intensity tilt effect at a specific angle,
    with the tilt effect ranging from -1 to 1 within the pupil.
    
    Parameters:
    - tiltx: The tilt in x direction.
    - tilty: The tilt in y direction.
    - tilt_angle: The angle of the intensity tilt in degrees.
    
    Returns:
    - The data with the intensity tilt applied.
    """
    
    # Convert angle to radians
    tilt_angle_rad = np.deg2rad(tilt_angle)


    # Tilt effect based on angle
    tilt_effect = (tiltx * np.cos(tilt_angle_rad) + 
                   tilty * np.sin(tilt_angle_rad))
    

    # Scale the tilt effect to [-1, 1] within the mask
    tilt_min = tilt_effect.min()
    tilt_max = tilt_effect.max()
    tilt_effect = 2 * (tilt_effect - tilt_min) / (tilt_max - tilt_min) - 1

    
    return tilt_effect/np.ptp(tilt_effect)

