import os
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils import *
from src.utils import set_data_dm
from src.dao_setup import init_setup
from .shm_loader import shm

slopes_image_shm = shm.slopes_image_shm

setup = init_setup()


def perform_push_pull_calibration_with_phase_basis(
    basis,
    phase_amp,
    ref_image,
    mask,
    bias_image,
    verbose=False,
    verbose_plot=False,
    mode_repetitions=None,
    push_pull=False,
    pull_push=True,
    **kwargs,
):
    """
    Perform push-pull calibration using a phase basis.

    Parameters:
    - ref_image: Reference image for normalization.
    - mask: Binary mask for normalization.
    - basis: Phase basis (e.g. Zernike or actuator basis).
    - phase_amp: Phase amplitude used for push/pull.
    - verbose: Print debug messages.
    - verbose_plot: Live plot during calibration.
    - mode_repetitions: Sequence or dict specifying how many times each mode is
      repeated and averaged. Modes with unspecified counts default to 1.
    - push_pull: If True, perform a push followed by a pull ([-phase_amp, phase_amp]).
    - pull_push: If True, perform a pull followed by a push ([phase_amp, -phase_amp]).
    - kwargs:
        - nact

    Returns:
    - pull_images, push_images, push_pull_images
    """
    # Load optional configuration values
    nact = kwargs.get("nact", setup.nact)
    camera_wfs = kwargs.get("camera_wfs", setup.camera_wfs)

    if not (push_pull or pull_push):
        raise ValueError("Either push_pull or pull_push must be True")
    
    # Set dimensions equal to img_size
    height, width = ref_image.shape

    # Number of phase modes
    nmodes_basis = basis.shape[0]

    # Normalize reference image with explicit bias_img set to zero
    normalized_reference_image = normalize_image(ref_image, mask, bias_image)

    # Prepare number of repetitions per mode
    if mode_repetitions is None:
        mode_repetitions = np.ones(nmodes_basis, dtype=int)
    elif isinstance(mode_repetitions, dict):
        reps = np.ones(nmodes_basis, dtype=int)
        for idx, rep in mode_repetitions.items():
            if 0 <= idx < nmodes_basis:
                reps[idx] = int(rep)
        mode_repetitions = reps
    else:
        mode_repetitions = np.asarray(mode_repetitions, dtype=int)
        if mode_repetitions.size == 1:
            mode_repetitions = np.full(nmodes_basis, mode_repetitions.item())
        elif mode_repetitions.size < nmodes_basis:
            reps = np.ones(nmodes_basis, dtype=int)
            reps[:mode_repetitions.size] = mode_repetitions
            mode_repetitions = reps
        elif mode_repetitions.size > nmodes_basis:
            mode_repetitions = mode_repetitions[:nmodes_basis]

    orders = []
    if push_pull:
        orders.append([-phase_amp, phase_amp])
    if pull_push:
        orders.append([phase_amp, -phase_amp])

    # Pre-allocate arrays
    pull_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)
    push_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)
    push_pull_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)

    # Start calibration
    start_time = time.time()
    
    # Setup for live display if verbose_plot is True
    if verbose_plot:
        plt.ion()  # Enable interactive mode
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for mode in range(nmodes_basis):

        rep_count = int(mode_repetitions[mode])

        # Initialize timing and accumulation arrays
        t0 = time.time()
        pull_acc = np.zeros((height, width), dtype=np.float32)
        push_acc = np.zeros((height, width), dtype=np.float32)
        pp_acc = np.zeros((height, width), dtype=np.float32)
        t1 = time.time()

        for i in range(rep_count):
            for order in orders:
                push_pull_img = np.zeros((height, width), dtype=np.float32)

                for amp in order:
                    # Compute Zernike phase pattern
                    t2 = time.time()
                    kl_mode = amp * basis[mode]
                    t3 = time.time()

                    # Send DM data of kl mode to the SLM
                    t4 = time.time()
                    set_data_dm(kl_mode, setup=setup,)
                    t7 = time.time()

                    # Capture the image and compute slopes
                    t8 = time.time()
                    n_frames=1
                    pyr_img = np.mean([camera_wfs.get_data() for i in range(n_frames)], axis=0)

                    slopes_image = get_slopes_image(
                        mask,
                        bias_image,
                        normalized_reference_image,
                        setup=setup,
                        pyr_img=pyr_img,
                    )

                    t9 = time.time()

                    # Store images for push & pull
                    t12 = time.time()
                    if amp > 0:
                        pull_acc += slopes_image / abs(amp)
                    else:
                        push_acc += slopes_image / abs(amp)

                    # Combine push-pull intensity
                    push_pull_img += slopes_image / (2 * amp)
                    t13 = time.time()

                # Save the combined response for this order
                pp_acc += push_pull_img

        divisor = rep_count * len(orders)
        # Average over repetitions and orders
        pull_images[mode, :, :] = pull_acc / divisor
        push_images[mode, :, :] = push_acc / divisor
        push_pull_images[mode, :, :] = pp_acc / divisor
        
        # Live display of images
        if verbose_plot:
            axs[0].imshow(pull_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(pull_images))
            axs[0].set_title('Pull Image')
            axs[0].axis('off')

            axs[1].imshow(push_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(push_images))
            axs[1].set_title('Push Image')
            axs[1].axis('off')

            axs[2].imshow(push_pull_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(push_pull_images))
            axs[2].set_title('Push-Pull Image')
            axs[2].axis('off')

            plt.draw()
            plt.pause(0.1)  # Pause to update the display

        if verbose and mode % 20 == 0:
            print(f"Processing mode: {mode}")
            print(f"Time to initialize arrays: {t1 - t0:.4f} s")
            print(f"Time to initialize phase mode: {t3 - t2:.4f} s")
            print(f"Time to send data to SLM: {t7 - t4:.4f} s")
            print(f"Time to capture image and compute slopes: {t9 - t8:.4f} s")
            print(f"Time to store and combine images: {t13 - t12:.4f} s")
            print("")

    # End calibration
    end_time = time.time()
    if verbose:
        print(f"Calibration completed in {end_time - start_time:.2f} seconds.")
    
    if verbose_plot:
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show the final display

    return pull_images, push_images, push_pull_images


def create_response_matrix(
    KL2Act,
    phase_amp,
    reference_image,
    mask,
    bias_image,
    *,
    verbose=True,
    verbose_plot=False,
    mode_repetitions=None,
    calibration_repetitions=1,
    push_pull=False,
    pull_push=True,
    **kwargs,
):
    """
    Run push-pull calibration and compute the response matrix.

    Parameters:
    -----------
    KL2Act : np.ndarray
        Actuator basis used for calibration.
    phase_amp : float
        Phase amplitude for push/pull.
    reference_image : np.ndarray
        Reference WFS image.
    mask : np.ndarray
        Mask for normalization and response extraction.
    folder_calib : str
        Path to folder for saving FITS files.
    nact : int
        Number of actuators (for filename metadata).
    verbose : bool
        If True, print debug info.
    verbose_plot : bool
        If True, show live calibration images.
    mode_repetitions : sequence, dict, or None
        Optional number of times to repeat and average each mode. Unspecified
        modes default to 1.
    calibration_repetitions : int
        Number of times to repeat the whole calibration process. The returned
        response matrices are the average over these runs.
    push_pull : bool
        If True, perform a push followed by a pull ([-phase_amp, phase_amp]).
    pull_push : bool
        If True, perform a pull followed by a push ([phase_amp, -phase_amp]).
    kwargs:
        - nact
        - folder_calib

    Returns:
    --------
    response_matrix_full : np.ndarray
        Flattened 2D response matrix for all modes.
    """
    
    nact = kwargs.get("nact", setup.nact)
    folder_calib = kwargs.get("folder_calib", setup.folder_calib)

    # Run push-pull calibration
    n_runs = max(1, int(calibration_repetitions))

    pull_sum = None
    push_sum = None
    pp_sum = None

    for run in range(n_runs):
        pull_images, push_images, push_pull_images = perform_push_pull_calibration_with_phase_basis(
            KL2Act,
            phase_amp,
            reference_image,
            mask,
            bias_image,
            verbose=verbose,
            verbose_plot=verbose_plot,
            mode_repetitions=mode_repetitions,
            push_pull=push_pull,
            pull_push=pull_push,
            **kwargs,
        )

        # Initialize accumulators using first run dimensions
        if pull_sum is None:
            pull_sum = np.zeros_like(pull_images, dtype=np.float32)
            push_sum = np.zeros_like(push_images, dtype=np.float32)
            pp_sum = np.zeros_like(push_pull_images, dtype=np.float32)

        pull_sum += pull_images
        push_sum += push_images
        pp_sum += push_pull_images


    pull_images = pull_sum / n_runs
    push_images = push_sum / n_runs
    push_pull_images = pp_sum / n_runs

    # Compute response matrices from the averaged push-pull images
    response_matrix_full = compute_response_matrix(push_pull_images).astype(np.float32)
    response_matrix_filtered = compute_response_matrix(push_pull_images, mask).astype(np.float32)

    # Define output filenames
    pull_filename     = f'processed_response_cube_KL2PWFS_only_pull_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
    push_filename     = f'processed_response_cube_KL2PWFS_only_push_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
    pushpull_filename = f'processed_response_cube_KL2PWFS_push-pull_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'

    # Save FITS files
    if verbose: print('Pull images saved')
    fits.writeto(os.path.join(folder_calib, pull_filename), pull_images, overwrite=True)

    if verbose: print('Push images saved')
    fits.writeto(os.path.join(folder_calib, push_filename), push_images, overwrite=True)

    if verbose: print('Push-pull images saved')
    fits.writeto(os.path.join(folder_calib, pushpull_filename), push_pull_images, overwrite=True)

    # Save the matrices
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filterd
    filtered = f'response_matrix_KL2S_filtered_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
    filtered_timestamped = f'response_matrix_KL2S_filtered_nact_{nact}_amp_{phase_amp}_3s_pyr_{timestamp}.fits'
    fits.writeto(os.path.join(folder_calib, filtered), response_matrix_filtered, overwrite=True)
    fits.writeto(os.path.join(folder_calib, filtered_timestamped), response_matrix_filtered, overwrite=True)
    if verbose: print(f"Filtered matrix saved to:\n  {filtered}\n  {filtered_timestamped}")
    
    # Full
    full = f'response_matrix_KL2S_full_nact_{nact}_amp_{phase_amp}_3s_pyr.fits'
    full_timestamped = f'response_matrix_KL2S_full_nact_{nact}_amp_{phase_amp}_3s_pyr_{timestamp}.fits'
    fits.writeto(os.path.join(folder_calib, full), response_matrix_full, overwrite=True)
    fits.writeto(os.path.join(folder_calib, full_timestamped), response_matrix_full, overwrite=True)
    if verbose: print(f"Full matrix saved to:\n  {full}\n  {full_timestamped}")

    return response_matrix_full, response_matrix_filtered


