import numpy as np
import time
import dao
from DEVICES_3.Basler_Pylon.test_pylon import *
import matplotlib.pyplot as plt
from src.utils import *
from src.dao_setup import *  # Import all variables from setup
import src.dao_setup as dao_setup


def perform_push_pull_calibration_with_phase_basis(basis, phase_amp, ref_image, mask, 
    verbose=False, verbose_plot=False, **kwargs):
    """
    Perform push-pull calibration using a phase basis.

    Parameters:
    - ref_image: Reference image for normalization.
    - mask: Binary mask for normalization.
    - basis: Phase basis (e.g. Zernike or actuator basis).
    - phase_amp: Phase amplitude used for push/pull.
    - verbose: Print debug messages.
    - verbose_plot: Live plot during calibration.
    - kwargs:
        - slm
        - camera
        - data_pupil_outer
        - data_pupil_inner
        - pupil_mask
        - small_pupil_mask
        - deformable_mirror

    Returns:
    - pull_images, push_images, push_pull_images
    """
    from src.dao_setup import wait_time  # lazy import to avoid circular dependency

    # Load devices and data from kwargs or fallback to dao_setup
    camera = kwargs.get("camera", dao_setup.camera_wfs)
    slm = kwargs.get("slm", dao_setup.slm)
    data_pupil_outer = kwargs.get("data_pupil_outer", dao_setup.data_pupil_outer)
    data_pupil_inner = kwargs.get("data_pupil_inner", dao_setup.data_pupil_inner)
    pupil_mask = kwargs.get("pupil_mask", dao_setup.pupil_mask)
    small_pupil_mask = kwargs.get("small_pupil_mask", dao_setup.small_pupil_mask)
    deformable_mirror = kwargs.get("deformable_mirror", dao_setup.deformable_mirror)

    # Set dimensions equal to img_size
    height, width = ref_image.shape

    # Number of phase modes
    nmodes_basis = basis.shape[0]
    
    # Compute size of the small pupil mask
    npix_small_pupil_grid = small_pupil_mask.shape[0]

    # Normalize reference image with explicit bias_img set to zero
    normalized_reference_image = normalize_image(ref_image, mask, bias_img=np.zeros_like(ref_image))

    # Pre-allocate arrays
    pull_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)
    push_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)
    push_pull_images = np.zeros((nmodes_basis, height, width), dtype=np.float32)
    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)

    # Start calibration
    start_time = time.time()
    
    # Setup for live display if verbose_plot is True
    if verbose_plot:
        plt.ion()  # Enable interactive mode
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for mode in range(nmodes_basis):

        # Initialize timing and temporary variables
        t0 = time.time()
        push_pull_img = np.zeros((height, width), dtype=np.float32)
        t1 = time.time()

        for amp in [-phase_amp, phase_amp]:
            # Compute Zernike phase pattern
            t2 = time.time()
            deformable_mirror.flatten()
            deformable_mirror.actuators = amp * basis[mode].reshape(deformable_mirror.num_actuators)
            data_dm[:, :] = deformable_mirror.opd.shaped
            t3 = time.time()

            # Add and wrap data within the pupil mask
            t4 = time.time()
            data_slm = compute_data_slm(data_dm=data_dm)
            slm.set_data(data_slm)
            time.sleep(wait_time)
            t7 = time.time()

            # Allow SLM to settle and capture the image
            t8 = time.time()
            pyr_img = camera.get_data()
            t9 = time.time()
            
            # Compute slopes
            normalized_pyr_img = normalize_image(pyr_img, mask, bias_img=np.zeros_like(pyr_img))
            slopes_image = compute_pyr_slopes(normalized_pyr_img, normalized_reference_image)

            # Store images for push & pull
            t10 = time.time()
            if amp > 0:
                pull_images[mode, :, :] = slopes_image / abs(amp)
            else:
                push_images[mode, :, :] = slopes_image / abs(amp)

            # Combine push-pull intensity
            push_pull_img += slopes_image / (2 * amp)
            t11 = time.time()

        # Save the combined response
        push_pull_images[mode, :, :] = push_pull_img
        
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

        if verbose and mode % 10 == 0:
            print(f"Processing mode: {mode}")
            print(f"Time to initialize arrays: {t1 - t0:.4f} s")
            print(f"Time to initialize phase mode: {t3 - t2:.4f} s")
            print(f"Time to send data to SLM: {t7 - t4:.4f} s")
            print(f"Time to capture image: {t9 - t8:.4f} s")
            print(f"Time to store and combine images: {t11 - t10:.4f} s")
            print("")

    # End calibration
    end_time = time.time()
    if verbose:
        print(f"Calibration completed in {end_time - start_time:.2f} seconds.")
    
    if verbose_plot:
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show the final display

    return pull_images, push_images, push_pull_images



# def perform_push_pull_calibration(slm, camera, img_size, deformable_mirror, phase_amp, data_pupil, pupil_mask, verbose=False, verbose_plot=False):
#     """
#     Perform push-pull calibration on the Zernike modes.
    
#     Parameters:
#     - slm: SLM instance.
#     - camera: Camera instance for capturing images.
#     - img_size: Tuple containing (height, width) for image size.
#     - deformable_mirror: Instance of the deformable mirror.
#     - phase_amp: Amplitude of the phase response.
#     - data_pupil: Pre-computed static pupil phase data for the SLM.
#     - pupil_mask: Binary mask defining the active pupil region on the SLM.
#     - verbose_plot: If True, enable live plotting.
#     - verbose: If True, enable printing status messages.

#     Returns:
#     - pull_images: Array of captured pull images.
#     - push_images: Array of captured push images.
#     - push_pull_images: Array of response images.
#     """
    
#     dataWidth = 1920
#     dataHeight = 1200
#     pixel_size_mm = 8e-3  # pixel size in mm 

#     # Set dimensions equal to img_size
#     height, width = img_size

#     # Number of DM modes
#     nmodes_dm = deformable_mirror.num_actuators  

#     # Pre-allocate arrays
#     pull_images = np.zeros((nmodes_dm, height, width), dtype=np.float32)
#     push_images = np.zeros((nmodes_dm, height, width), dtype=np.float32)
#     push_pull_images = np.zeros((nmodes_dm, height, width), dtype=np.float32)
#     data_dm = np.zeros((dataHeight, dataWidth), dtype=np.float32)

#     # Start calibration
#     start_time = time.time()
#     camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

#     # Setup for live display if verbose_plot is True
#     if verbose_plot:
#         plt.ion()  # Enable interactive mode
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     for mode in range(nmodes_dm):

#         # Initialize timing and temporary variables
#         push_pull_img = np.zeros((height, width), dtype=np.float32)

#         for amp in [-phase_amp, phase_amp]:
#             # Apply phase amp to an actuator
#             deformable_mirror.flatten()
#             deformable_mirror.actuators[mode] = amp
#             data_dm[:, :] = deformable_mirror.surface.shaped

#             # Add and wrap data within the pupil mask
#             data_slm = ((data_dm + data_pupil)) 
#             data_slm[pupil_mask] = ((data_slm[pupil_mask] * 256) % 256)

#             # Send the phase pattern to the SLM
#             slm.set_data(data_slm.astype(np.uint8))

#             # Allow SLM to settle and capture the image
#             time.sleep(0.3)
#             grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#             if grabResult.GrabSucceeded():
#                 img = np.array(grabResult.Array.astype("float64"))
#             grabResult.Release()

#             # Crop image using img_size directly
#             img = img#[:height, :width]

#             # Store images for push & pull
#             if amp > 0:
#                 pull_images[mode, :, :] = img / abs(amp)
#             else:
#                 push_images[mode, :, :] = img / abs(amp)

#             # Combine push-pull intensity
#             push_pull_img += img * (1 / amp)

#         # Save the combined response
#         push_pull_images[mode, :, :] = push_pull_img
        

#         # Live display of images
#         if verbose_plot:
#             axs[0].imshow(pull_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(pull_images))
#             axs[0].set_title('Pull Image')
#             axs[0].axis('off')

#             axs[1].imshow(push_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(push_images))
#             axs[1].set_title('Push Image')
#             axs[1].axis('off')

#             axs[2].imshow(push_pull_images[mode, :, :], cmap='gray', vmin=0, vmax=np.max(push_pull_images))
#             axs[2].set_title('Push-Pull Image')
#             axs[2].axis('off')

#             plt.draw()
#             plt.pause(0.1)  # Pause to update the display

#         if verbose and mode % 10 == 0:
#             print(f"Processing mode: {mode}")

#     # End calibration
#     end_time = time.time()
#     if verbose:
#         print(f"Calibration completed in {end_time - start_time:.2f} seconds.")

#     # Stop camera grabbing
#     camera.StopGrabbing()
    
#     if verbose_plot:
#         plt.ioff()  # Disable interactive mode
#         plt.show()  # Show the final display

#     return pull_images, push_images, push_pull_images

