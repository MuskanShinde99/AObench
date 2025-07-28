# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:57:48 2024

@author: RISTRETTO
"""

import os
import time
import numpy as np
from astropy.io import fits
from scipy.ndimage import center_of_mass
from scipy.interpolate import interp2d

from .shm_loader import shm

DEFAULT_SETUP = None

def set_default_setup(setup):
    """Register a default setup used when none is provided."""
    global DEFAULT_SETUP
    DEFAULT_SETUP = setup




def compute_data_slm(data_dm=None, data_phase_screen=0, data_dm_flat=0, setup=None, **kwargs):
    """
    Computes the SLM data by combining phase screen and deformable mirror data
    with pupil-related masks and arrays.

    Parameters:
    - data_dm (float or ndarray, optional): Deformable mirror phase data. If ``None``, ``setup.data_dm`` is used.
    - data_phase_screen (float or ndarray): Phase screen data to add (default: 0).
    - **kwargs:
        - data_pupil_inner (ndarray): Inner pupil data array.
        - data_pupil_outer (ndarray): Outer pupil data array.
        - pupil_mask (ndarray): Boolean mask for pupil over the full slm.
        - small_pupil_mask (ndarray): Boolean mask for small square pupil on the slm.

    Returns:
    - data_slm (ndarray): Resulting SLM data.
    """
    
    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP

    if data_dm is None:
        data_dm = setup.data_dm
    data_pupil_inner = kwargs.get("data_pupil_inner", setup.data_pupil_inner)
    data_pupil_outer = kwargs.get("data_pupil_outer", setup.data_pupil_outer)
    pupil_mask = kwargs.get("pupil_mask", setup.pupil_mask)
    small_pupil_mask = kwargs.get("small_pupil_mask", setup.small_pupil_mask)

    data_slm = data_pupil_outer.copy()
    data_inner = (((data_pupil_inner + data_dm + data_phase_screen) * 256) % 256)
    data_slm[pupil_mask] = data_inner[small_pupil_mask]

    return data_slm.astype(np.uint8)


def set_dm_actuators(dm, actuators=None, dm_flat=None, setup=None):
    """Set DM actuators and update the shared memory grid."""

    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP

    if actuators is None:
        actuators = np.zeros(setup.nact ** 2)
    if dm_flat is None:
        dm_flat = setup.dm_flat

    actuators = np.asarray(actuators)
    if actuators.size != setup.nact ** 2:
        raise ValueError(
            f"Expected {setup.nact ** 2} actuators, got {actuators.size}"
        )

    dm.actuators = actuators + dm_flat

    dm_act_shm = shm.dm_act_shm
    dm_act_shm.set_data(np.asarray(dm.actuators).reshape(setup.nact, setup.nact))
    

# ---------------------------------------------------------------------------
def set_data_dm(actuators=None, *, setup=None, dm_flat=None, **kwargs):
    """Flatten the DM, optionally apply ``actuators`` and show the resulting
    phase on the SLM.

    Parameters
    ----------
    actuators : array_like, optional
        Actuator values to apply. Defaults to zeros.
    dm_flat : array_like, optional
        Flat map added to the actuators. Defaults to ``setup.dm_flat``.
    setup : object, optional
        DAO setup providing defaults for devices and dimensions.
    kwargs : optional
        ``dm``                 – DM instance. Defaults to ``setup.deformable_mirror``.
        ``slm``                – SLM instance. Defaults to ``setup.slm``.
        ``npix_small_pupil_grid`` – DM grid size. Defaults to ``setup.npix_small_pupil_grid``.
        ``pupil_setup``        – Setup for ``compute_data_slm``. Defaults to ``setup.pupil_setup``.
        ``wait_time``          – Delay after sending data. Defaults to ``setup.wait_time`` or ``0``.

    Returns
    -------
    tuple of ndarray
        ``(data_dm, data_slm)`` arrays that were sent to the SLM.
    """

    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP

    slm = kwargs.get("slm", getattr(setup, "slm", None))
    if slm is None:
        raise ValueError("SLM instance must be provided")

    npix_small_pupil_grid = kwargs.get(
        "npix_small_pupil_grid",
        getattr(setup, "npix_small_pupil_grid", None),
    )
    if npix_small_pupil_grid is None:
        raise ValueError("npix_small_pupil_grid must be provided")

    wait_time = kwargs.get("wait_time", getattr(setup, "wait_time", 0))

    dm = kwargs.get("dm", getattr(setup, "deformable_mirror", None))
    if dm is None:
        raise ValueError("Deformable mirror instance must be provided")

    dm.flatten()
    set_dm_actuators(dm, actuators, dm_flat=dm_flat, setup=setup)

    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
    data_dm[:, :] = dm.opd.shaped / 2

    pupil_setup = kwargs.get("pupil_setup", getattr(setup, "pupil_setup", setup))
    data_slm = compute_data_slm(data_dm=data_dm, setup=pupil_setup)
    slm.set_data(data_slm)
    time.sleep(wait_time)

    return data_dm, data_slm




def create_psf_mask(psf, crop_size=100, radius=50):
    """
    Creates a mask for the PSF image by cropping a region around the center,
    computing the center of mass, and creating a circular mask around it.

    Parameters:
    - psf (numpy array): The PSF image to process.
    - crop_size (int): The size of the region to crop around the center. Default is 100.
    - radius (int): The radius of the circular mask for integration. Default is 50 pixels.

    Returns:
    - psf_mask (numpy array): The mask for the PSF.
    - psf_center (numpy array): The center coordinates of the PSF.
    """
    # Define crop size for centering
    center_x, center_y = psf.shape[0] // 2, psf.shape[1] // 2
    x_start, x_end = center_x - crop_size // 2, center_x + crop_size // 2
    y_start, y_end = center_y - crop_size // 2, center_y + crop_size // 2
    
    # Crop the PSF
    cropped_psf = psf[x_start:x_end, y_start:y_end]
    
    # Compute center of mass
    psf_center = np.array(center_of_mass(cropped_psf))
    
    # Adjust to original coordinates
    psf_center += [x_start, y_start]
    print(f'Center of PSF: {psf_center}')
    
    # Define a small region around the peak for integration (circular mask)
    y, x = np.indices(psf.shape)
    psf_mask = (x - psf_center[1])**2 + (y - psf_center[0])**2 <= radius**2
    
    # # Visualize the mask
    # plt.figure()
    # plt.imshow(psf_mask)
    # plt.colorbar()
    # plt.title('Mask to compute Strehl')
    # plt.show()
    
    return psf_mask, psf_center



def bias_correction(image, bias_image):
    """
    Subtracts the bias image from the given image.

    Parameters:
    - image (numpy array): The input image to be corrected.
    - bias_image (numpy array): The bias image to subtract from the input image.

    Returns:
    - numpy array: The corrected image after bias subtraction.
    """
    
    # Ensure the images are numpy arrays
    image = np.asarray(image)
    bias_image = np.asarray(bias_image)
    
    # Check if the image and bias image have the same shape
    if image.shape != bias_image.shape:
        raise ValueError("The image and bias image must have the same shape.")
    
    # Subtract the bias image from the input image
    corrected_image = image - bias_image
    
    return corrected_image


def normalize_image(image, mask, bias_img=None):
    """
    Normalizes the masked image by applying the mask, and then dividing by the absolute sum of the masked image.
    
    Parameters:
    - image (numpy array): The image to normalize.
    - mask (numpy array): The mask to apply to the image.
    - bias_img (numpy array, optional): The bias image to subtract from the image. Defaults to an array of zeros if not provided.
    
    Returns:
    - numpy array: The normalized image.
    """
    
    # If bias_img is not provided, initialize it as an array of zeros
    if bias_img is None:
        bias_img = np.zeros_like(image)

    # Apply bias correction to image
    bias_corrected_image = bias_correction(image, bias_img)
    
    # Apply the mask to the image
    masked_image = bias_corrected_image * mask
    
    # Normalize the masked image by dividing by the absolute sum of the masked image
    normalized_image = masked_image / np.abs(np.sum(masked_image))

    return normalized_image


def get_slopes_image(mask, bias_image, normalized_reference_image, pyr_img=None, setup=None, **kwargs):
    """Capture a PyWFS frame and compute its slope image.

    The resulting slopes image is always written to the global
    ``slopes_img_shm`` shared memory.

    Parameters
    ----------
    camera_wfs : object
        Camera used to grab a new frame when ``pyr_img`` is ``None``.
    mask : numpy.ndarray
        Processing mask applied to the raw WFS image.
    bias_image : numpy.ndarray
        Bias image subtracted from the captured frame.
    normalized_reference_image : numpy.ndarray
        Normalized reference image used for slope computation.
    pyr_img : numpy.ndarray, optional
        Pre-acquired pyramid image. If provided, ``camera_wfs`` is not queried.

    Returns
    -------
    numpy.ndarray
        The computed slopes image.
    """

    slopes_img_shm = shm.slopes_img_shm

    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP
    
    camera_wfs = kwargs.get("camera_wfs", setup.camera_wfs)

    if pyr_img is None:
        pyr_img = camera_wfs.get_data()

    normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
    slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)
    # print('slopes_image data type', slopes_image.dtype)
    # print('slopes_image shape', slopes_image.shape)
    slopes_img_shm.set_data(slopes_image)
    return slopes_image


def compute_pyr_slopes(normalized_pyr_img, normalized_ref_img):
    """
    Computes the slopes between a pyramid image and a reference image.

    Parameters:
    - pyr_img (numpy array): The normalized pyramid image to process.
    - ref_img (numpy array): The normalized reference image.
    
    Returns:
    - numpy array: The slope between the processed pyramid image and the processed reference image.
    """

    # Compute the slope (difference) between the normalized pyramid and reference images
    slopes_image = normalized_pyr_img - normalized_ref_img
    
    return slopes_image

# Function to check if a matrix file exists in a specified folder
def matrix_exists(folder, filename):
    """
    Checks whether a matrix file exists in the given folder.

    Parameters:
    - folder (str): The folder path where the file is supposed to be.
    - filename (str): The name of the file to check for existence.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    # Join the folder path and filename to get the full file path
    file_path = os.path.join(folder, filename)
    
    # Use os.path.exists() to check if the file exists at the specified path
    return os.path.exists(file_path)


def bin_matrix_2d(matrix, new_rows, new_cols):
    """Interpolates a matrix to a new shape using 2D interpolation."""
    old_rows, old_cols = matrix.shape
    old_x = np.linspace(0, 1, old_cols)  # Original column scale
    old_y = np.linspace(0, 1, old_rows)  # Original row scale
    new_x = np.linspace(0, 1, new_cols)  # New column scale
    new_y = np.linspace(0, 1, new_rows)  # New row scale
    
    interp_func = interp2d(old_x, old_y, matrix, kind='linear')  # 2D interpolation
    interpolated_matrix = interp_func(new_x, new_y)
    
    return interpolated_matrix

def process_response_images_3s_pyr(images, mask, ref_image, bias_image):
    """
    Compute the processed response cube

    Parameters:
    - images: A 3D array of images (data cube).
    - mask: Mask array for cropping.
    - ref_image: Reference image of the pwfs.

    Returns:
    - response_matrix: The computed response matrix.
    """
    # Get dimensions of the data cube
    num_images, height, width = images.shape
    print('Number of images:', num_images)
    
    # Multiply by mask and normalize the reference image
    normalized_reference_image = normalize_image(ref_image, mask, bias_image)

    # Initialize a list to hold the new data cube images
    processed_images = []

    # Process each image in the data cube
    for i in range(num_images):
        # Multiply by the mask and  Normalize by the sum of the image
        normalized_image = normalize_image(images[i], mask, bias_image)

        # calculate slope image
        slope_image = compute_pyr_slopes(normalized_image, normalized_reference_image)

        # Add the processed slope image to the list
        processed_images.append(slope_image)

    # Convert the list of processed images into an array
    processed_images = np.array(processed_images)
    
    return processed_images


def process_response_images(images, mask, normalized_reference_image, coms):
    """
    Compute the processed response cube

    Parameters:
    - images: A 3D array of images (data cube).
    - mask: Mask array for cropping.
    - normalized_reference_image: Reference image for normalization.
    - coms: Centers of mass for cropping.

    Returns:
    - response_matrix: The computed response matrix.
    """
    # Get dimensions of the data cube
    num_images, height, width = images.shape
    print('Number of images:', num_images)

    # Initialize a list to hold the new data cube images
    processed_images = []

    # Process each image in the data cube
    for i in range(num_images):
        # Multiply by the mask
        masked_image = images[i] * mask

        # Normalize by the sum of the image
        normalized_image = masked_image / np.abs(np.sum(masked_image))

        # Subtract the reference image
        normalized_image = normalized_image - normalized_reference_image

        # Combine the four cropped regions into one image
        cropped_image = crop_and_combine(normalized_image, coms, crop_size=40)

        # Add the processed image to the list
        processed_images.append(cropped_image)

    # Convert the list of processed images into an array
    processed_images = np.array(processed_images)
    return processed_images

def find_pupil_coms(mask):
    """
    Finds the center of mass for each of the four quadrants of the mask created for pyramid images.

    Parameters:
    mask (2D): 2D array of mask

    Returns:
    tuple: A tuple containing the center of mass coordinates for each pupil.
    """
    # Print
    #print(f"Mask dimensions: {mask.shape}")

    # Divide the image into 4 parts (quadrants)
    height, width = mask.shape
    mid_y, mid_x = height // 2, width // 2

    # Define the four quadrants
    quadrant_1 = mask[:mid_y, :mid_x]  # Top-left
    quadrant_2 = mask[:mid_y, mid_x:]  # Top-right
    quadrant_3 = mask[mid_y:, :mid_x]  # Bottom-left
    quadrant_4 = mask[mid_y:, mid_x:]  # Bottom-right

    # Find the center of mass for each quadrant
    com_1 = center_of_mass(quadrant_1)
    com_2 = center_of_mass(quadrant_2)
    com_3 = center_of_mass(quadrant_3)
    com_4 = center_of_mass(quadrant_4)

    # Adjust center of mass coordinates relative to the original image
    com_2 = (com_2[0], com_2[1] + mid_x)  # Adjust x for the second quadrant
    com_3 = (com_3[0] + mid_y, com_3[1])  # Adjust y for the third quadrant
    com_4 = (com_4[0] + mid_y, com_4[1] + mid_x)  # Adjust both for the fourth quadrant

    return com_1, com_2, com_3, com_4


# Define function to crop a region around a given center of mass
def crop_around_com(mask, com, crop_size=50):
    y, x = com
    half_size = crop_size // 2
    height, width = mask.shape
    start_y = int(max(y - half_size, 0))
    end_y = int(min(y + half_size, height))
    start_x = int(max(x - half_size, 0))
    end_x = int(min(x + half_size, width))
    
    return mask[start_y:end_y, start_x:end_x]


def crop_and_combine(image, coms, crop_size=50):
    """
    Crop regions around each center of mass and combine them into one image.

    Parameters:
    - image: 2D numpy array representing the image.
    - coms: List of tuples, where each tuple contains the (y, x) center of mass coordinates.
    - crop_size: The size of the crop region around each center of mass.

    Returns:
    - combined_image: 2D numpy array representing the combined cropped image.
    """
    
    # Ensure we have exactly 4 center of mass coordinates
    if len(coms) != 4:
        raise ValueError("coms must contain exactly 4 center of mass coordinates")

    crops = []
    for com in coms:
        y, x = com
        
        # Compute half the crop size
        half_size = crop_size // 2
        
        # Get the dimensions of the original image
        height, width = image.shape
        
        # Calculate the starting and ending coordinates for the crop
        start_y = int(max(y - half_size, 0))  # Ensure we don't go below 0
        end_y = int(min(y + half_size, height))  # Ensure we don't go beyond the image height
        start_x = int(max(x - half_size, 0))  # Ensure we don't go below 0
        end_x = int(min(x + half_size, width))  # Ensure we don't go beyond the image width
        
        # Crop the region from the image and append to the list of crops
        crops.append(image[start_y:end_y, start_x:end_x])
    
    # Calculate dimensions for the combined image
    crop_height, crop_width = crops[0].shape
    combined_height = crop_height * 2
    combined_width = crop_width * 2
    
    # Create an empty combined image
    combined_image = np.zeros((combined_height, combined_width), dtype=image.dtype)
    
    # Place each cropped image in the correct position
    combined_image[:crop_height, :crop_width] = crops[0]
    combined_image[:crop_height, crop_width:] = crops[1]
    combined_image[crop_height:, :crop_width] = crops[2]
    combined_image[crop_height:, crop_width:] = crops[3]
    
    return combined_image



"""
# Example usage
folder = os.path.join('C:\\', 'Users', 'RISTRETTO', 'RISTRETTO_AO_bench_images', 'Push-pull_calibration')
pup_size = 2  # Define pupil size in mm 
mask_filename = f'mask_pup_{pup_size}mm.fits'
mask_data = fits.getdata(os.path.join(folder, mask_filename))

coms = find_pupil_coms(mask_data)
print("Center of Mass for each quadrant:")
print(f"Quadrant 1: {coms[0]}")
print(f"Quadrant 2: {coms[1]}")
print(f"Quadrant 3: {coms[2]}")
print(f"Quadrant 4: {coms[3]}")

"""


