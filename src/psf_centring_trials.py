#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:49:52 2025

@author: laboptic
"""

from pypylon import pylon
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import time
from hcipy import *
from autograd import grad
import autograd.numpy as anp  # Use autograd's NumPy
from DEVICES_3.Basler_Pylon.test_pylon import *

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

import dao_setup

# Access devices
slm = dao_setup.slm
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp
camera_wfs.Open()

def capture_image():
    """Captures an image from the camera."""
    return pylonGrab(camera_wfs, 1)

def compute_pupil_intensities(img, pupil_coords, radius):
    """Computes the average intensity for each pupil."""
    intensities = []
    for (x, y) in pupil_coords:
        mask = np.zeros_like(img)
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        mask_area = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        intensities.append(np.mean(img[mask_area]))
        
    print('intensities:', np.array(intensities))
    return np.array(intensities)

# def cost_function(values):
#     return anp.sqrt(anp.mean((values - anp.mean(values))**2))

# def cost_function(intensities, delta=1.0):
#     intensities = anp.array(intensities)
#     mean_intensity = anp.mean(intensities)
#     residuals = anp.abs(intensities - mean_intensity)
#     return anp.mean(anp.where(residuals < delta, 0.5 * residuals ** 2, delta * (residuals - 0.5 * delta)))


# def cost_function(intensities):
#     """Computes the entropy-based cost function for pupil intensities."""
#     intensities = anp.array(intensities)
#     p_i = intensities / anp.sum(intensities)  # Normalize to create a probability distribution
#     p_i = anp.clip(p_i, 1e-10, 1)  # Avoid log(0) issues
#     return -anp.sum(p_i * anp.log(p_i))  # Use autograd.numpy.log




def update_amplitudes(amplitudes, gradients, learning_rate=0.001):
    """Updates amplitudes using gradient descent."""
    return amplitudes - learning_rate * gradients

def compute_gradients(intensities):
    """Computes gradients using automatic differentiation from Autograd."""
    gradient_func = grad(cost_function)
    return gradient_func(intensities)


def gradient_descent(initial_amplitudes, fixed_index=2, max_iters=200, tol=1e-3):
    """Performs gradient descent to balance light intensities using an entropy-based cost function."""
    amplitudes = np.array(initial_amplitudes)
    
    for i in range(max_iters):
        data_pupil = dao_setup.update_pupil(new_ttf_amplitudes=amplitudes)
        slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))
        time.sleep(0.3)  # Allow time for the update

        img = capture_image()
        pupil_coords = [(1938.50395876, 1582.96110293), 
                        (1712.19789503, 1590.92554125), 
                        (1827.06954403, 1778.60921125)]  # Example positions
        radius = 85
        intensities = compute_pupil_intensities(img, pupil_coords, radius)
        loss = cost_function(intensities)

        print(f"Iteration {i+1}, Loss: {loss:.4f}, Amplitudes: {amplitudes}")
        
        if loss < tol:
            break

        gradients = compute_gradients(intensities)
        #print('gradients:', gradients)

        # Update only the first two amplitudes, keep the third fixed
        amplitudes[:fixed_index] = update_amplitudes(amplitudes[:fixed_index], gradients[:fixed_index])
    
    return amplitudes

# Initial TTF amplitudes; the third term is fixed during optimization
new_ttf_amplitudes = [-0.3, 0.3, 0.4]  # Keep 0.4 fixed
optimized_amplitudes = gradient_descent(new_ttf_amplitudes)
print(f"Optimized Amplitudes: {optimized_amplitudes}")

# Capture final image
final_img = capture_image()
plt.figure()
plt.imshow(final_img, cmap='gray')
plt.title('Final Balanced Pupil Intensities')
plt.colorbar()
plt.show()
