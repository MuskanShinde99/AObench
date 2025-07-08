
from pypylon import pylon
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from hcipy import *
from skopt import gp_minimize
from DEVICES_3.Basler_Pylon.test_pylon import *
import sys
from pathlib import Path
import cv2
import re

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

from src.dao_setup import *  # Import all variables from setup
#import src.dao_setup as dao_setup


# Access the SLM and cameras

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
    data_pupil = update_pupil(new_ttf_amplitudes=[*amplitudes, fixed_amplitude])
    slm.set_data(((data_pupil * 256) % 256).astype(np.uint8))  # Update SLM data
    time.sleep(wait_time)  # Wait for the update to take effect

    # Capture and average 5 images
    num_images = 5
    images = [camera_wfs.get_data() for i in range(num_images)]
    img = np.mean(images, axis=0)
    
    #Compute pupil intensities
    intensities = compute_pupil_intensities(img, pupil_coords, radius)  # Compute intensities
    
    # Calculate the variance of the intensities as the cost
    mean_intensity = np.mean(intensities)
    variance = np.mean((intensities - mean_intensity) ** 2)
    
    # Print iteration number and variance
    print(f"Iteration: {iteration} | Variance: {round(variance)} | Tipi-tilt amptitude: {amplitudes}")

    
    # Check stopping condition: if variance is less than 5
    if variance < 5:
        print(f"Stopping condition met: Variance is below threshold.")
        stop_optimization = True  # Set stopping flag
        
    return variance + 1e-3  # Add a small noise floor


# Global list to store cost function values for debugging
cost_values = []

def optimize_amplitudes(initial_amplitudes, pupil_coords, radius,
                        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
                        n_calls=200):
    """
    Optimize amplitudes using Scikit-Optimize's gp_minimize. 
    Bayesian optimization using Gaussian Processes.

    Parameters:
    - initial_amplitudes (list): Initial guess for tip and tilt amplitudes
    - pupil_coords (list): List of (x, y) pupil center coordinates
    - radius (float): Approximate radius of pupil mask
    - bounds (list): Search bounds for tip and tilt amplitudes
    - n_calls (int): Maximum number of optimization iterations

    Returns:
    - result.x (list): Optimized [tip, tilt] amplitudes
    """

    # Wrapper around the cost function to track the cost at each iteration
    def wrapped_cost_function(amps):
        iteration = wrapped_cost_function.iteration
        wrapped_cost_function.iteration += 1
        cost = cost_function(amps, pupil_coords, radius, iteration)
        cost_values.append(cost)  # Append cost at each iteration for debugging
        return cost

    # Initialize iteration counter
    wrapped_cost_function.iteration = 0

    # Callback to allow early stopping if condition is met
    def stop_callback(res):
        """Callback to stop optimization early if condition is met."""
        return stop_optimization  # Check global flag

    # Perform Gaussian Process optimization
    result = gp_minimize(
        wrapped_cost_function,
        bounds,           # Search bounds
        n_calls=n_calls,  # Number of evaluations
        random_state=42,  # Reproducibility
        callback=[stop_callback]  # Add callback for early stopping
    )

    return result.x  # Return optimized amplitudes



def center_psf_on_pyramid_tip(mask,
                              initial_tt_amplitudes=[-0.5, 0.2],
                              focus=[0.4],
                              bounds=[(-1.0, 1.0), (-1.0, 1.0)],
                              n_calls=200,
                              update_setup_file=False,
                              verbose=False,
                              verbose_plot=False):
    """
    Optimize tip-tilt amplitudes to balance pupil intensities using a given binary mask.

    Parameters:
    - mask (np.ndarray): Binary mask of pupil regions
    - initial_tt_amplitudes (list): Initial [tip, tilt] guess
    - focus (list): Single-element list with focus value (e.g., [0.4])
    - bounds (list of tuple): Bounds for the [tip, tilt] amplitudes
    - n_calls (int): Number of iterations for the optimizer
    - update_setup_file (bool): If True, update `ttf_amplitudes` in dao_setup.py
    - verbose (bool): Print processing info
    - verbose_plot (bool): Show final image and optimization cost plot

    Returns:
    - new_ttf_amplitudes (list): Optimized [tip, tilt, focus] amplitudes
    """
    
    global stop_optimization, cost_values
    # Reset global state so subsequent calls start a fresh optimization
    stop_optimization = False
    cost_values = []

    # Ensure mask is binary and of correct dtype
    mask = mask.astype(np.uint8)

    # Find connected components in the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    pupil_centers = centroids[1:]  # Skip background label
    radii = [np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) for i in range(1, num_labels)]
    radius = int(round(np.mean(radii)))  # Average radius

    if verbose:
        print(f"Found {len(pupil_centers)} pupils")
        print(f"Average pupil radius: {radius:.2f}")

    # Optimize tip-tilt amplitudes using Bayesian optimization
    optimized_tt_amplitudes = optimize_amplitudes(
        initial_tt_amplitudes,
        pupil_centers,
        radius,
        bounds=bounds,
        n_calls=n_calls
    )

    # Append the fixed focus amplitude to complete the vector
    new_ttf_amplitudes = optimized_tt_amplitudes + focus

    if verbose:
        print(f"Optimized Tip-Tilt-Focus Amplitudes: {new_ttf_amplitudes}")

    # Capture final image after optimization
    final_img = camera_wfs.get_data()

    # Example of plotting the cost function values after optimization
    if verbose_plot and 'cost_values' in globals():
        plt.figure()
        plt.plot(cost_values)
        plt.title("Cost Function Values Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Variance")
        plt.show()

    # Display Final Image
    if verbose_plot:
        plt.figure()
        plt.imshow(final_img, cmap='gray')
        plt.title('Final Balanced Pupil Intensities')
        plt.colorbar()
        plt.show()

    # Update the dao_setup.py file with the new optimized amplitudes
    if update_setup_file:
        dao_setup_path = Path(ROOT_DIR) / 'src/dao_setup.py'

        # Read the existing file content
        with open(dao_setup_path, 'r') as file:
            content = file.read()

        # Update the file with the new optimized amplitudes
        new_line = f"ttf_amplitudes = {new_ttf_amplitudes}"
        updated_content = re.sub(r"ttf_amplitudes\s*=\s*\[.*?\]", new_line, content)

        # Write the updated content back to the setup file
        with open(dao_setup_path, 'w') as file:
            file.write(updated_content)

        if verbose:
            print("✅ Updated `ttf_amplitudes` in dao_setup.py")
    else:
        if verbose:
            print("⚠️ Skipped updating `ttf_amplitudes` in `dao_setup.py`")

    return new_ttf_amplitudes



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