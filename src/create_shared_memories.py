import dao
import numpy as np
import os

from src.config import config

ROOT_DIR = config.root_dir

setup = None

small_pupil_mask_shm = None
pupil_mask_shm = None
slopes_img_shm = None
dm_act_shm = None
KL2Act_shm = None
KL2Phs_shm = None
valid_pixels_mask_shm = None
bias_image_shm = None
reference_psf_shm = None
normalized_ref_psf_shm = None
reference_image_shm = None
normalized_ref_image_shm = None
npix_valid_shm = None
delay_shm = None
gain_shm = None
leakage_shm = None
num_iterations_shm = None
slopes_image_shm = None
phase_screen_shm = None
dm_phase_shm = None
phase_residuals_shm = None
normalized_psf_shm = None
commands_shm = None
residual_modes_shm = None
computed_modes_shm = None
dm_kl_modes_shm = None


def init_shared_memories(current_setup):
    """Create shared memory segments using the given setup."""
    global setup, small_pupil_mask_shm, pupil_mask_shm, slopes_img_shm, dm_act_shm
    global KL2Act_shm, KL2Phs_shm, valid_pixels_mask_shm, bias_image_shm
    global reference_psf_shm, normalized_ref_psf_shm, reference_image_shm
    global normalized_ref_image_shm, npix_valid_shm, delay_shm, gain_shm
    global leakage_shm, num_iterations_shm, slopes_image_shm, phase_screen_shm
    global dm_phase_shm, phase_residuals_shm, normalized_psf_shm, commands_shm
    global residual_modes_shm, computed_modes_shm, dm_kl_modes_shm

    setup = current_setup

    small_pupil_mask_shm = dao.shm(
        '/tmp/small_pupil_mask.im.shm',
        np.zeros((setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float32)
    )
    pupil_mask_shm = dao.shm(
        '/tmp/pupil_mask.im.shm',
        np.zeros((setup.dataHeight, setup.dataWidth), dtype=np.float32)
    )

    slopes_img_shm = dao.shm(
        '/tmp/slopes_img.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.uint32)
    )

    dm_act_shm = dao.shm(
        '/tmp/dm_act.im.shm',
        np.zeros((setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float64)
    )

    KL2Act_shm = dao.shm(
        '/tmp/KL2Act.im.shm',
        np.zeros((setup.nmodes_KL, setup.nact**2), dtype=np.float64)
    )
    KL2Phs_shm = dao.shm(
        '/tmp/KL2Phs.im.shm',
        np.zeros((setup.nmodes_KL, setup.npix_small_pupil_grid**2), dtype=np.float64)
    )

    valid_pixels_mask_shm = dao.shm(
        '/tmp/valid_pixels_mask.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.uint8)
    )
    bias_image_shm = dao.shm(
        '/tmp/bias_image.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.float64)
    )
    reference_psf_shm = dao.shm(
        '/tmp/reference_psf.im.shm',
        np.zeros((setup.img_size_fp_cam, setup.img_size_fp_cam), dtype=np.uint16)
    )
    normalized_ref_psf_shm = dao.shm(
        '/tmp/normalized_ref_psf.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.float64)
    )
    reference_image_shm = dao.shm(
        '/tmp/reference_image.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.uint16)
    )
    normalized_ref_image_shm = dao.shm(
        '/tmp/normalized_ref_image.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.float64)
    )
    npix_valid_shm = dao.shm(
        '/tmp/npix_valid.im.shm', np.zeros((1, 1), dtype=np.uint32)
    )

    delay_shm = dao.shm('/tmp/delay.im.shm', np.zeros((1, 1), dtype=np.uint32))
    gain_shm = dao.shm('/tmp/gain.im.shm', np.zeros((1, 1), dtype=np.float32))
    leakage_shm = dao.shm(
        '/tmp/leakage.im.shm', np.zeros((1, 1), dtype=np.float32)
    )
    num_iterations_shm = dao.shm(
        '/tmp/num_iterations.im.shm', np.zeros((1, 1), dtype=np.uint32)
    )

    slopes_image_shm = dao.shm(
        '/tmp/slopes_image.im.shm',
        np.zeros((setup.img_size_wfs_cam, setup.img_size_wfs_cam), dtype=np.float64)
    )
    phase_screen_shm = dao.shm(
        '/tmp/phase_screen.im.shm',
        np.zeros((setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float32)
    )
    dm_phase_shm = dao.shm(
        '/tmp/dm_phase.im.shm',
        np.zeros((setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float32)
    )
    phase_residuals_shm = dao.shm(
        '/tmp/phase_residuals.im.shm',
        np.zeros((setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), dtype=np.float32)
    )
    normalized_psf_shm = dao.shm(
        '/tmp/normalized_psf.im.shm',
        np.zeros((setup.img_size_fp_cam, setup.img_size_fp_cam), dtype=np.float64)
    )
    commands_shm = dao.shm(
        '/tmp/commands.im.shm',
        np.zeros((setup.nmodes_dm, 1), dtype=np.float32)
    )
    residual_modes_shm = dao.shm(
        '/tmp/residual_modes.im.shm',
        np.zeros((setup.nmodes_KL, 1), dtype=np.float32)
    )
    computed_modes_shm = dao.shm(
        '/tmp/computed_modes.im.shm',
        np.zeros((setup.nmodes_KL, 1), dtype=np.float32)
    )
    dm_kl_modes_shm = dao.shm(
        '/tmp/dm_kl_modes.im.shm',
        np.zeros((setup.nmodes_KL, 1), dtype=np.float32)
    )


__all__ = [name for name in globals() if not name.startswith('_')]
