from matplotlib import pyplot as plt
import numpy as np
from skopt import gp_minimize
from pathlib import Path
import cv2
import re

from src.config import config
from src.dao_setup import set_data_dm

ROOT_DIR = config.root_dir

from src.dao_setup import init_setup, ROOT_DIR


setup = init_setup()
wait_time = setup.wait_time
pupil_setup = setup.pupil_setup
camera_wfs = setup.camera_wfs


# Access the SLM and cameras


def compute_pupil_intensities(img, pupil_coords, radius):
    """Compute average intensity for each pupil location."""
    intensities = []
    for x, y in pupil_coords:
        mask = np.zeros_like(img)
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        mask_area = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2  # Create a circular mask
        intensities.append(
            np.mean(img[mask_area])
        )  # Calculate mean intensity in the mask area

    return np.array(intensities)


# Global stopping flag
stop_optimization = False


def cost_function(amplitudes, pupil_coords, radius, iteration, variance_threshold=5):
    """Cost function for optimization based on intensity variance.

    Parameters
    ----------
    amplitudes : list
        Current tip and tilt amplitudes.
    pupil_coords : list
        List of pupil centre coordinates.
    radius : float
        Radius for the pupil mask.
    iteration : int
        Current optimisation iteration.
    variance_threshold : float, optional
        Variance value below which the optimisation should stop. Defaults to ``5``.
    """
    global stop_optimization  # Flag to stop when pupil intensities are equal

    pupil_setup.update_pupil(tt_amplitudes=amplitudes)
    # Apply the updated flat map to the DM and SLM
    set_data_dm(setup=setup)

    # Capture and average 5 images
    num_images = 5
    images = [camera_wfs.get_data() for i in range(num_images)]
    img = np.mean(images, axis=0)

    # Compute pupil intensities
    intensities = compute_pupil_intensities(
        img, pupil_coords, radius
    )  # Compute intensities

    # Calculate the variance of the intensities as the cost
    mean_intensity = np.mean(intensities)
    normalized_intensities = intensities / mean_intensity
    variance = np.mean((normalized_intensities - 1) ** 2)

    # Print iteration number and variance
    print(
        f"Iteration: {iteration} | Variance: {variance:.3f} | Tipi-tilt amptitude: {amplitudes}"
    )

    # Check stopping condition based on the provided threshold
    if variance < variance_threshold:
        print(
            f"Stopping condition met: Variance {variance:.3f} below threshold {variance_threshold}."
        )
        stop_optimization = True  # Set stopping flag

    return variance + 1e-3  # Add a small noise floor


# Global list to store cost function values for debugging
cost_values = []


def optimize_amplitudes(
    pupil_coords,
    radius,
    bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    n_calls=200,
    variance_threshold=5,
):
    """
    Optimize amplitudes using Scikit-Optimize's gp_minimize.
    Bayesian optimization using Gaussian Processes.

    Parameters:
    - pupil_coords (list): List of (x, y) pupil center coordinates
    - radius (float): Approximate radius of pupil mask
    - bounds (list): Search bounds for tip and tilt amplitudes
    - n_calls (int): Maximum number of optimization iterations
    - variance_threshold (float): Stopping threshold for the variance

    Returns:
    - result.x (list): Optimized [tip, tilt] amplitudes
    """

    # Wrapper around the cost function to track the cost at each iteration
    def wrapped_cost_function(amps):
        iteration = wrapped_cost_function.iteration
        wrapped_cost_function.iteration += 1
        cost = cost_function(
            amps,
            pupil_coords,
            radius,
            iteration,
            variance_threshold=variance_threshold,
        )
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
        bounds,  # Search bounds
        n_calls=n_calls,  # Number of evaluations
        random_state=42,  # Reproducibility
        callback=[stop_callback],  # Add callback for early stopping
    )

    return result.x  # Return optimized amplitudes


def center_psf_on_pyramid_tip(
    mask,
    bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    n_calls=200,
    update_setup_file=False,
    verbose=False,
    verbose_plot=False,
    variance_threshold=5,
):
    """
    Optimize tip-tilt amplitudes to balance pupil intensities using a given binary mask.

    Parameters:
    - mask (np.ndarray): Binary mask of pupil regions
    - bounds (list of tuple): Bounds for the [tip, tilt] amplitudes
    - n_calls (int): Number of iterations for the optimizer
    - update_setup_file (bool): If True, update `tt_amplitudes` in dao_setup.py
    - verbose (bool): Print processing info
    - verbose_plot (bool): Show final image and optimization cost plot
    - variance_threshold (float): Stop optimization when variance drops below this value

    Returns:
    - new_tt_amplitudes (list): Optimized [tip, tilt] amplitudes
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
        pupil_centers,
        radius,
        bounds=bounds,
        n_calls=n_calls,
        variance_threshold=variance_threshold,
    )

    new_tt_amplitudes = optimized_tt_amplitudes

    if verbose:
        print(f"Optimized Tip-Tilt Amplitudes: {new_tt_amplitudes}")

    # Capture final image after optimization
    final_img = camera_wfs.get_data()

    # Example of plotting the cost function values after optimization
    if verbose_plot and "cost_values" in globals():
        plt.figure()
        plt.plot(cost_values)
        plt.title("Cost Function Values Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Variance")
        plt.show()

    # Display Final Image
    if verbose_plot:
        plt.figure()
        plt.imshow(final_img, cmap="gray")
        plt.title("Final Balanced Pupil Intensities")
        plt.colorbar()
        plt.show()

    # Update the dao_setup.py file with the new optimized amplitudes
    if update_setup_file:
        dao_setup_path = Path(ROOT_DIR) / "src/dao_setup.py"

        # Read the existing file content
        with open(dao_setup_path, "r") as file:
            content = file.read()

        # Update the file with the new optimized amplitudes
        new_line = f"tt_amplitudes = {new_tt_amplitudes}"
        updated_content = re.sub(r"tt_amplitudes\s*=\s*\[.*?\]", new_line, content)

        # Write the updated content back to the setup file
        with open(dao_setup_path, "w") as file:
            file.write(updated_content)

        if verbose:
            print("Updated `tt_amplitudes` in dao_setup.py")
    else:
        if verbose:
            print("Skipped updating `tt_amplitudes` in `dao_setup.py`")

    return new_tt_amplitudes


# %%
"""
# Example
# Define pupil coordinates and radius
pupil_coords = [(1938.50, 1582.96), 
                (1712.20, 1590.93), 
                (1827.07, 1778.61)]
radius = 83

optimized_amplitudes = optimize_amplitudes(
    pupil_coords,
    radius,
    variance_threshold=5,
)  # Optimize the amplitudes
print(f"Optimized Amplitudes: {optimized_amplitudes}")

# Capture and display the final image after optimization
final_img = camera_wfs.get_data()  # Capture final image
plt.figure()
plt.imshow(final_img, cmap='gray')
plt.title('Final Balanced Pupil Intensities')
plt.colorbar()
plt.show()"""
