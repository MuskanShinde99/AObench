
from pypylon import pylon
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from hcipy import *
from skopt import gp_minimize
from DEVICES_3.Basler_Pylon.test_pylon import *

# Set working directory
os.chdir('/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO')

import src.dao_setup as dao_setup

# Access the SLM and cameras
slm = dao_setup.slm
camera_wfs = dao_setup.camera_wfs
camera_fp = dao_setup.camera_fp

def compute_pupil_intensities(img, pupil_coords, radius):
    """Compute average intensity for each pupil location."""
    intensities = []
    for (x, y) in pupil_coords:
        mask = np.zeros_like(img)
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        mask_area = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2  # Create a circular mask
        intensities.append(np.mean(img[mask_area]))  # Calculate mean intensity in the mask area
        
    return np.array(intensities)

# Global stopping flag
stop_optimization = False  

def cost_function(amplitudes, pupil_coords, radius, iteration):
    """Cost function for optimization based on intensity variance."""
    global stop_optimization  #Flag to stop when pupil intensities are equal

    fixed_amplitude = 0.4  # Keep the third amplitude fixed
    data_pupil = dao_setup.update_pupil(new_ttf_amplitudes=[*amplitudes, fixed_amplitude])
    slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))  # Update SLM data
    time.sleep(dao_setup.wait_time)  # Wait for the update to take effect

    img = camera_wfs.get_data()  # Capture an image after updating
    intensities = compute_pupil_intensities(img, pupil_coords, radius)  # Compute intensities
    
    # Print iteration number and intensities
    print(f"Iteration {iteration}: Pupil Intensities: {intensities}")

    # Check if rounded intensities are equal
    rounded_intensities = np.round(intensities, 2).astype(int)
    if np.all(rounded_intensities == rounded_intensities[0]):
        print("Stopping condition met: Pupil intensities are equal.")
        stop_optimization = True  # Set stopping flag

    # Calculate the variance of the intensities as the cost
    mean_intensity = np.mean(intensities)
    variance = np.mean((intensities - mean_intensity) ** 2)
    return variance + 1e-3  # Add a small noise floor


# Global list to store cost function values for debugging
cost_values = []

def optimize_amplitudes(initial_amplitudes, pupil_coords, radius):
    """Optimize amplitudes using Scikit-Optimize's gp_minimize. 
    Bayesian optimization using Gaussian Processes."""
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]  # Set bounds for the first two amplitudes

    # Wrapper around the cost function to track the cost at each iteration
    def wrapped_cost_function(amps):
        iteration = wrapped_cost_function.iteration
        wrapped_cost_function.iteration += 1
        cost = cost_function(amps, pupil_coords, radius, iteration)
        cost_values.append(cost)  # Append cost at each iteration for debugging
        return cost

    # Initialize iteration counter
    wrapped_cost_function.iteration = 0

    def stop_callback(res):
        """Callback to stop optimization early if condition is met."""
        return stop_optimization  # Check global flag

    result = gp_minimize(
        wrapped_cost_function,
        bounds,
        n_calls=300,
        random_state=42,
        callback=[stop_callback]  # Add callback for early stopping
    )

    return result.x  # Return optimized amplitudes




#%%
"""
# Example
# Define pupil coordinates and radius
pupil_coords = [(1938.50, 1582.96), 
                (1712.20, 1590.93), 
                (1827.07, 1778.61)]
radius = 83

# Initial TTF amplitudes; third term is fixed
new_ttf_amplitudes = [-0.5, 0.7]
optimized_amplitudes = optimize_amplitudes(new_ttf_amplitudes, pupil_coords, radius)  # Optimize the amplitudes
print(f"Optimized Amplitudes: {optimized_amplitudes + [0.4]}")  # Include fixed amplitude in the output

# Capture and display the final image after optimization
final_img = camera_wfs.get_data()  # Capture final image
plt.figure()
plt.imshow(final_img, cmap='gray')
plt.title('Final Balanced Pupil Intensities')
plt.colorbar()
plt.show()
"""