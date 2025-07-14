# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:49:32 2024

@author: RISTRETTO
"""

from matplotlib import pyplot as plt
import numpy as np
from hcipy import *
from src.tilt_functions import *

def create_slm_circular_pupil(tilt_amp_outer, tilt_amp_inner, pupil_size, pupil_mask, slm):
    
    dataWidth = 1920
    dataHeight = 1200
    pixel_size_mm = 8e-3  #  pixel size in mm
    npix_pupil = pupil_size/pixel_size_mm 
    
    # Create a pupil grid 
    x = np.linspace(-dataWidth / 2, dataWidth / 2, dataWidth)
    y = np.linspace(-dataHeight / 2, dataHeight / 2, dataHeight)
    xx, yy = np.meshgrid(x, y)
    
    # Reserve memory for the data:
    data_outer = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_inner = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    
    # Compute the outer data with tilt
    tilt_outer = tilt_amp_outer * (yy / np.ptp(yy)) # Normalize by ptp amplitudwe
    data_outer = tilt_outer%1
    # Apply the pupil mask to the data:
    data_outer[pupil_mask] = 0

    #Create an inner pupil with tilt
    tilt_inner = tilt_amp_inner * apply_intensity_tilt(pupil_mask, npix_pupil / 2, tilt_angle=0)
    data_inner = tilt_inner
    
    #Combine outer and inner data
    data = data_inner + data_outer
    
    return data

def create_slm_circular_pupil_horizontal(blaze_period_outer, blaze_period_inner, pupil_size, pupil_mask, slm):
    """
    Generates a blazed grating pattern with a circular pupil mask for an SLM.

    Parameters:
    -----------
    blaze_period_outer : int or float
        The period of the outer blazed grating in pixels.
        
    blaze_period_inner : int or float
        The period of the inner blazed grating in pixels.
        
    pupil_size : float
        The diameter of the circular pupil in millimeters (mm).
        
    slm : object
        An object containing properties of the spatial light modulator (SLM), including width_px, height_px, and pixelsize_m.

    Returns:
    --------
    data : 2D numpy array
        The generated 2D array containing the combined blazed grating pattern with a circular pupil mask applied.
    """
    dataWidth = 1920
    dataHeight = 1200
    pixel_size_mm = 8e-3 # pixel size in mm 
    
    # Reserve memory for the data:
    data_outer = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_inner = np.zeros((dataHeight, dataWidth), dtype=np.float32)

    # Calculate the outer blazed grating:
    for y in range(dataHeight):
        for x in range(dataWidth):
            data_outer[y, x] = float(x % blaze_period_outer) / blaze_period_outer

    # Calculate the inner blazed grating:
    for y in range(dataHeight):
        for x in range(dataWidth):
            data_inner[y, x] = float(x % blaze_period_inner) / blaze_period_inner


    # Apply the mask to the data:
    data_inner[pupil_mask] = 1 - data_inner[pupil_mask]
    data_outer[pupil_mask] = 0
    data_inner[~pupil_mask] = 0

    # Combine the data:
    data = data_outer + data_inner

    return data

def create_slm_circular_pupil_vertical(blaze_period_outer, blaze_period_inner, pupil_size, pupil_mask, slm):
    """
    Generates a blazed grating pattern with a circular pupil mask for an SLM.

    Parameters:
    -----------
    blaze_period_outer : int or float
        The period of the outer blazed grating in pixels.
        
    blaze_period_inner : int or float
        The period of the inner blazed grating in pixels.
        
    pupil_size : float
        The diameter of the circular pupil in millimeters (mm).
        
    slm : object
        An object containing properties of the spatial light modulator (SLM), including width_px, height_px, and pixelsize_m.

    Returns:
    --------
    data : 2D numpy array
        The generated 2D array containing the combined blazed grating pattern with a circular pupil mask applied.
    """
    
    dataWidth = 1920
    dataHeight = 1200
    pixel_size_mm = 8e-3 # pixel size in mm 
    
    # Reserve memory for the data:
    data_outer = np.zeros((dataHeight, dataWidth), dtype=np.float32)
    data_inner = np.zeros((dataHeight, dataWidth), dtype=np.float32)

    # Calculate the outer blazed grating:
    for y in range(dataHeight):
        for x in range(dataWidth):
            data_outer[y, x] = float(y % blaze_period_outer) / blaze_period_outer

    # Calculate the inner blazed grating:
    for y in range(dataHeight):
        for x in range(dataWidth):
            data_inner[y, x] = float(x % blaze_period_inner) / blaze_period_inner

    # Apply the mask to the data:
    data_inner[pupil_mask] = 1 - data_inner[pupil_mask]
    data_outer[pupil_mask] = 0
    data_inner[~pupil_mask] = 0

    # Combine the data:
    data = data_outer + data_inner

    return data


# Example usage:
"""
blaze_period_outer = 5
blaze_period_inner = 10
pupil_size = 2  # [mm]
slm_width_px =  1920
slm_height_px =  1200
pixel_size_mm = 8e-3  #  pixel size in mm

data_slm = create_slm_circular_pupil(blaze_period_outer, blaze_period_inner, pupil_size, slm_width_px, slm_height_px, pixel_size_mm)

# Show data on SLM:
error = slm.showData(data_slm)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
"""

def create_slm_circular_pupil_with_tilts(blaze_period_outer, tilt_amp, pupil_size, pupil_mask, slm, tilt_parameters):
    """
    Create a circular pupil on the SLM and apply multiple tilts based on input angles and amplitudes.

    Parameters:
    -----------
    blaze_period_outer : int
        Blaze period for the outer region of the pupil.
    blaze_period_inner : int
        Blaze period for the inner region of the pupil.
    pupil_size : float
        Size of the pupil (in mm).
    slm : object
        An object containing properties of the spatial light modulator (SLM), including width_px, height_px, and pixelsize_m.
    tilt_parameters : list of tuples
        Each tuple contains (tilt_angle, tilt_amplitude).

    Returns:
    --------
    data_with_tilts : numpy.ndarray
        The SLM data with added tilts.
    combined_tilts : numpy.ndarray
        The combined tilt field.
    """
    
    dataWidth = 1920
    dataHeight = 1200
    pixel_size_mm = 8e-3 # pixel size in mm 

    # Validate that tilt_parameters contains valid tuples of angles and amplitudes
    if not all(isinstance(t, tuple) and len(t) == 2 for t in tilt_parameters):
        raise ValueError("tilt_parameters must be a list of tuples with (angle, amplitude).")

    # Calculate the number of pixels in the pupil (diameter in pixels)
    Npix = pupil_size / pixel_size_mm

    # Create circular pupil data on SLM (assume this function is provided)
    data = create_slm_circular_pupil_new(blaze_period_outer, tilt_amp, pupil_size, pupil_mask, slm)

    # Create empty tilt field (the same size as the SLM)
    data_tilt = np.zeros((dataHeight, dataWidth), dtype=np.float32)

    # Initialize combined tilt field (with same shape as the aperture)
    combined_tilts = np.zeros_like(mask).astype(np.float64)

    # Apply tilts using the input tuples (tilt_angle, tilt_amplitude)
    for angle, amplitude in tilt_parameters:
        tilt = apply_intensity_tilt(pupil_mask, Npix / 2, tilt_angle=angle)
        combined_tilts += amplitude * tilt

    # Add the combined tilt to the original SLM data
    data_with_tilts = data + combined_tilts

    return data_with_tilts



# Example usage
"""
# Define tilt parameters
tilt_parameters = [
    (-45, 0.2),  # Tilt of -45 degrees with amplitude 0.2
    (0, -2.8),   # Tilt of 0 degrees with amplitude -2.8
    (-90, -0.5), # Tilt of -90 degrees with amplitude -0.5
    (45, 0.0)    # Tilt of 45 degrees with amplitude 0.0
]

# Call the function
data_with_tilts, combined_tilts = create_slm_circular_pupil_with_tilts(blaze_period_outer, blaze_period_inner, pupil_size, slm_width_px, slm_height_px, pixel_size_mm, tilt_parameters)

# Added tilt
plt.figure()
plt.imshow(data_with_tilts)
plt.title('Data with Added Tilt in Pupil')
plt.colorbar()
plt.show()

# Show data on SLM:
error = slm.showData(data_with_tilts)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
"""