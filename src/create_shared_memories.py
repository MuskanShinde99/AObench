import dao
import numpy as np
import os
import toml

from src.config import config
ROOT_DIR = config.root_dir

with open("setup_params.toml", "r") as f:
    config = toml.load(f)

# nact = config["geneva"]["nact"]
# img_size_wfs_cam_x = config["geneva"]["img_size_wfs_cam_x"]
# img_size_wfs_cam_y = config["geneva"]["img_size_wfs_cam_y"]
# img_size_fp_cam_x = config["geneva"]["img_size_fp_cam_x"]
# img_size_fp_cam_y = config["geneva"]["img_size_fp_cam_y"]
# npix_small_pupil_grid = config["geneva"]["npix_small_pupil_grid"]
# nmodes_KL = config["geneva"]["nmodes_KL"]
# nmodes_dm = config["geneva"]["nmodes_dm"]

nact = config["ohp"]["nact"]
img_size_wfs_cam_x = config["ohp"]["img_size_wfs_cam_x"]
img_size_wfs_cam_y = config["ohp"]["img_size_wfs_cam_y"]
img_size_fp_cam_x = config["ohp"]["img_size_fp_cam_x"]
img_size_fp_cam_y = config["ohp"]["img_size_fp_cam_y"]
npix_small_pupil_grid = config["ohp"]["npix_small_pupil_grid"]
nmodes_KL = config["ohp"]["nmodes_KL"]
nmodes_dm = config["ohp"]["nmodes_dm"]



#%%
# Pupil / Grids
# small_pupil_mask_shm = dao.shm('/tmp/small_pupil_mask.im.shm', np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
# pupil_mask_shm       = dao.shm('/tmp/pupil_mask.im.shm',       np.zeros((dataHeight, dataWidth), dtype=np.float32))

# WFS
slopes_img_shm       = dao.shm('/tmp/slopes_img.im.shm',       np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.float64))

# Deformable Mirror
dm_act_shm           = dao.shm('/tmp/dm_act.im.shm',           np.zeros((nact, nact), dtype=np.float64))

# SLM
# slm_shm             = dao.shm('/tmp/slm.im.shm',              np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))

# Transformation Matrices
KL2Act_shm           = dao.shm('/tmp/KL2Act.im.shm',           np.zeros((nmodes_KL, nact**2), dtype=np.float64))
KL2Phs_shm           = dao.shm('/tmp/KL2Phs.im.shm',           np.zeros((nmodes_KL, npix_small_pupil_grid**2), dtype=np.float64))
# Act2Phs_shm        = dao.shm('/tmp/Act2Phs.im.shm',          np.zeros((nact**2, npix_small_pupil_grid**2), dtype=np.float64))
# Phs2Act_shm        = dao.shm('/tmp/Phs2Act.im.shm',          np.zeros((npix_small_pupil_grid**2, nact**2), dtype=np.float64))
# Act2KL_shm         = dao.shm('/tmp/Act2KL.im.shm',           np.zeros((nact**2, nmodes_KL), dtype=np.float64))
# Phs2KL_shm         = dao.shm('/tmp/Phs2KL.im.shm',           np.zeros((npix_small_pupil_grid**2, nmodes_KL), dtype=np.float64))

# Zernike 
# Znk2Act_shm        = dao.shm('/tmp/Znk2Act.im.shm',          np.zeros((nmode_Znk, nact**2), dtype=np.float32))
# Act2Znk_shm        = dao.shm('/tmp/Act2Znk.im.shm',          np.zeros((nact**2, nmode_Znk), dtype=np.float32))
# Znk2Phs_shm        = dao.shm('/tmp/Znk2Phs.im.shm',          np.zeros((nmode_Znk, npix_small_pupil_grid**2), dtype=np.float32))
# Phs2Znk_shm        = dao.shm('/tmp/Phs2Znk.im.shm',          np.zeros((npix_small_pupil_grid**2, nmode_Znk), dtype=np.float32))


# Calibration / Reference
valid_pixels_mask_shm        = dao.shm('/tmp/valid_pixels_mask.im.shm',        np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.uint8))
bias_image_shm               = dao.shm('/tmp/bias_image.im.shm',               np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.float32))
reference_image_shm          = dao.shm('/tmp/reference_image.im.shm',          np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.uint16))
normalized_ref_image_shm     = dao.shm('/tmp/ref_img_norm.im.shm',     np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.float64))
reference_psf_shm            = dao.shm('/tmp/reference_psf.im.shm',            np.zeros((img_size_fp_cam_x, img_size_fp_cam_y), dtype=np.uint16))
normalized_ref_psf_shm       = dao.shm('/tmp/normalized_ref_psf.im.shm',       np.zeros((img_size_fp_cam_x, img_size_fp_cam_y), dtype=np.float64))

npix_valid_shm               = dao.shm('/tmp/npix_valid.im.shm',               np.zeros((1, 1), dtype=np.uint32))

# KL ↔ Slopes
# KL2PWFS_cube_shm           = dao.shm('/tmp/KL2PWFS_cube.im.shm',             np.zeros((nmodes_KL, img_size_wfs_cam_x*img_size_wfs_cam_x_y), dtype=np.float64))
# slopes_shm                 = dao.shm('/tmp/slopes.im.shm',                   np.zeros((npix_valid, 1), dtype=np.uint32))
# KL2S_shm                   = dao.shm('/tmp/KL2S.im.shm',                     np.zeros((nmodes_KL, npix_valid), dtype=np.float64))
# S2KL_shm                   = dao.shm('/tmp/S2KL.im.shm',                     np.zeros((npix_valid, nmodes_KL), dtype=np.float64))

# Control Parameters
delay_set_shm                = dao.shm('/tmp/delay_set.im.shm',                   np.zeros((1, 1), dtype=np.uint32))
gain_shm                     = dao.shm('/tmp/gain.im.shm',                    np.zeros((1, 1), dtype=np.float32))
leakage_shm                  = dao.shm('/tmp/leakage.im.shm',                 np.zeros((1, 1), dtype=np.float32))
num_iterations_shm           = dao.shm('/tmp/num_iterations.im.shm',          np.zeros((1, 1), dtype=np.uint32))

# AO loop plots
slopes_image_shm             = dao.shm('/tmp/slopes_image.im.shm',            np.zeros((img_size_wfs_cam_x, img_size_wfs_cam_y), dtype=np.float64))
phase_screen_shm             = dao.shm('/tmp/phase_screen.im.shm',            np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
dm_phase_shm                 = dao.shm('/tmp/dm_phase.im.shm',                np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
phase_residuals_shm          = dao.shm('/tmp/phase_residuals.im.shm',         np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
normalized_psf_shm           = dao.shm('/tmp/normalized_psf.im.shm',          np.zeros((img_size_fp_cam_x, img_size_fp_cam_y), dtype=np.float64))
commands_shm                 = dao.shm('/tmp/commands.im.shm',                np.zeros((nmodes_dm, 1), dtype=np.float32))
residual_modes_shm           = dao.shm('/tmp/residual_modes.im.shm',          np.zeros((nmodes_KL, 1), dtype=np.float32))
computed_modes_shm           = dao.shm('/tmp/computed_modes.im.shm',          np.zeros((nmodes_KL, 1), dtype=np.float32))
dm_kl_modes_shm              = dao.shm('/tmp/dm_kl_modes.im.shm',             np.zeros((nmodes_KL, 1), dtype=np.float32))
norm_flux_pyr_img_shm        = dao.shm('/tmp/norm_flux_pyr_img.im.shm',       np.zeros((1, 1), dtype=np.float32))
strehl_ratio_shm             = dao.shm('/tmp/strehl_ratio.im.shm',            np.zeros((1, 1), dtype=np.float32))
