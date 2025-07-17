import dao
import numpy as np
import os
import sys
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Explicit imports from dao_setup
from src.dao_setup import (npix_small_pupil_grid, dataHeight, dataWidth, 
                           img_size_wfs_cam, img_size_fp_cam, 
                            nmodes_KL, nact, nmodes_dm)

# Pupil / Grids
small_pupil_mask_shm = dao.shm('/tmp/small_pupil_mask.im.shm', np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
pupil_mask_shm       = dao.shm('/tmp/pupil_mask.im.shm',       np.zeros((dataHeight, dataWidth), dtype=np.float32))

# WFS
slopes_img_shm       = dao.shm('/tmp/slopes_img.im.shm',       np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.uint32))

# Deformable Mirror
dm_act_shm           = dao.shm('/tmp/dm_act.im.shm',           np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))

# SLM
# slm_shm             = dao.shm('/tmp/slm.im.shm',              np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))

# Transformation Matrices
# Act2Phs_shm        = dao.shm('/tmp/Act2Phs.im.shm',          np.zeros((nact**2, npix_small_pupil_grid**2), dtype=np.float64))
# Phs2Act_shm        = dao.shm('/tmp/Phs2Act.im.shm',          np.zeros((npix_small_pupil_grid**2, nact**2), dtype=np.float64))
KL2Act_shm           = dao.shm('/tmp/KL2Act.im.shm',           np.zeros((nmodes_KL, nact**2), dtype=np.float64))
# Act2KL_shm         = dao.shm('/tmp/Act2KL.im.shm',           np.zeros((nact**2, nmodes_KL), dtype=np.float64))
KL2Phs_shm           = dao.shm('/tmp/KL2Phs.im.shm',           np.zeros((nmodes_KL, npix_small_pupil_grid**2), dtype=np.float64))
# Phs2KL_shm         = dao.shm('/tmp/Phs2KL.im.shm',           np.zeros((npix_small_pupil_grid**2, nmodes_KL), dtype=np.float64))

# Zernike 
# Znk2Act_shm        = dao.shm('/tmp/Znk2Act.im.shm',          np.zeros((nmode_Znk, nact**2), dtype=np.float32))
# Act2Znk_shm        = dao.shm('/tmp/Act2Znk.im.shm',          np.zeros((nact**2, nmode_Znk), dtype=np.float32))
# Znk2Phs_shm        = dao.shm('/tmp/Znk2Phs.im.shm',          np.zeros((nmode_Znk, npix_small_pupil_grid**2), dtype=np.float32))
# Phs2Znk_shm        = dao.shm('/tmp/Phs2Znk.im.shm',          np.zeros((npix_small_pupil_grid**2, nmode_Znk), dtype=np.float32))

# Calibration / Reference
valid_pixels_mask_shm        = dao.shm('/tmp/valid_pixels_mask.im.shm',        np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.uint8))
bias_image_shm               = dao.shm('/tmp/bias_image.im.shm',               np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.float64))
reference_psf_shm            = dao.shm('/tmp/reference_psf.im.shm',            np.zeros((img_size_fp_cam, img_size_fp_cam), dtype=np.uint16))
normalized_ref_psf_shm       = dao.shm('/tmp/normalized_ref_psf.im.shm',       np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.float64))
reference_image_shm          = dao.shm('/tmp/reference_image.im.shm',          np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.uint16))
normalized_ref_image_shm     = dao.shm('/tmp/normalized_ref_image.im.shm',     np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.float64))
npix_valid_shm               = dao.shm('/tmp/npix_valid.im.shm',               np.zeros((1, 1), dtype=np.uint32))

# KL ↔ Slopes
# KL2PWFS_cube_shm           = dao.shm('/tmp/KL2PWFS_cube.im.shm',             np.zeros((nmodes_KL, img_size_wfs_cam**2), dtype=np.float64))
# slopes_shm                 = dao.shm('/tmp/slopes.im.shm',                   np.zeros((npix_valid, 1), dtype=np.uint32))
# KL2S_shm                   = dao.shm('/tmp/KL2S.im.shm',                     np.zeros((nmodes_KL, npix_valid), dtype=np.float64))
# S2KL_shm                   = dao.shm('/tmp/S2KL.im.shm',                     np.zeros((npix_valid, nmodes_KL), dtype=np.float64))

# Control Parameters
delay_shm                    = dao.shm('/tmp/delay.im.shm',                   np.zeros((1, 1), dtype=np.uint32))
gain_shm                     = dao.shm('/tmp/gain.im.shm',                    np.zeros((1, 1), dtype=np.float32))
leakage_shm                  = dao.shm('/tmp/leakage.im.shm',                 np.zeros((1, 1), dtype=np.float32))
num_iterations_shm           = dao.shm('/tmp/num_iterations.im.shm',          np.zeros((1, 1), dtype=np.uint32))

# AO loop plots
slopes_image_shm             = dao.shm('/tmp/slopes_image.im.shm',            np.zeros((img_size_wfs_cam, img_size_wfs_cam), dtype=np.float64))
phase_screen_shm             = dao.shm('/tmp/phase_screen.im.shm',            np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
dm_phase_shm                 = dao.shm('/tmp/dm_phase.im.shm',                np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
phase_residuals_shm          = dao.shm('/tmp/phase_residuals.im.shm',         np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32))
normalized_psf_shm           = dao.shm('/tmp/normalized_psf.im.shm',          np.zeros((img_size_fp_cam, img_size_fp_cam), dtype=np.float64))
commands_shm                 = dao.shm('/tmp/commands.im.shm',                np.zeros((nmodes_dm, 1), dtype=np.float32))
residual_modes_shm           = dao.shm('/tmp/residual_modes.im.shm',          np.zeros((nmodes_KL, 1), dtype=np.float32))
computed_modes_shm           = dao.shm('/tmp/computed_modes.im.shm',          np.zeros((nmodes_KL, 1), dtype=np.float32))
dm_kl_modes_shm              = dao.shm('/tmp/dm_kl_modes.im.shm',             np.zeros((nmodes_KL, 1), dtype=np.float32))
