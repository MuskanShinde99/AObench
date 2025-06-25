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

import src.dao_setup as dao_setup  # Import the setup file

nact = dao_setup.nact_valid
nmodes_dm = dao_setup.nact_valid
nmodes_KL = dao_setup.nact_valid
nmode_Znk = dao_setup.nact_valid
pupil_size = dao_setup.pupil_size
npix_pupil = dao_setup.npix_pupil
npix_small_pupil_grid = dao_setup.npix_small_pupil_grid
dataHeight = dao_setup.dataHeight
dataWidth = dao_setup.dataWidth
img_size = dao_setup.img_size


# Pupil / Grids
small_pupil_mask_shm = dao.shm('/tmp/small_pupil_mask.im.shm', np.zeros((npix_small_pupil_grid, npix_small_pupil_grid)).astype(np.float32)) 
pupil_mask = dao.shm('/tmp/pupil_mask.im.shm', np.zeros((dataHeight, dataWidth)).astype(np.float32)) 

# WFS
slopes_img_shm = dao.shm('/tmp/slopes_img.im.shm', np.zeros((img_size, img_size)).astype(np.uint32)) 

# Deformable Mirror
dm_act_shm = dao.shm('/tmp/dm_act.im.shm', np.zeros((npix_small_pupil_grid, npix_small_pupil_grid)).astype(np.float32)) 

# Transformation Matrices
Act2Phs_shm = dao.shm('/tmp/Act2Phs.im.shm', np.zeros((nact**2, npix_small_pupil_grid**2)).astype(np.float32)) 
Phs2Act_shm = dao.shm('/tmp/Phs2Act.im.shm', np.zeros((npix_small_pupil_grid**2, nact**2)).astype(np.float32)) 

KL2Act_shm = dao.shm('/tmp/KL2Act.im.shm', np.zeros((nmodes_KL,nact**2)).astype(np.float32)) 
Act2KL_shm = dao.shm('/tmp/Act2KL.im.shm', np.zeros((nact**2, nmodes_KL)).astype(np.float32)) 
KL2Phs_shm = dao.shm('/tmp/KL2Phs.im.shm', np.zeros((nmodes_KL, npix_small_pupil_grid**2)).astype(np.float32)) 
Phs2KL_shm = dao.shm('/tmp/Phs2KL.im.shm', np.zeros((npix_small_pupil_grid**2, nmodes_KL)).astype(np.float32)) 

Znk2Act_shm = dao.shm('/tmp/Znk2Act.im.shm', np.zeros((nmode_Znk,nact**2)).astype(np.float32)) 
Act2Znk_shm = dao.shm('/tmp/Act2Znk.im.shm', np.zeros((nact**2, nmode_Znk)).astype(np.float32)) 
Znk2Phs_shm = dao.shm('/tmp/Znk2Phs.im.shm', np.zeros((nmode_Znk, npix_small_pupil_grid**2)).astype(np.float32)) 
Phs2Znk_shm = dao.shm('/tmp/Phs2Znk.im.shm', np.zeros((npix_small_pupil_grid**2, nmode_Znk)).astype(np.float32)) 


# Calibration / Reference
bias_image_shm = dao.shm('/tmp/bias_image.im.shm', np.zeros((img_size, img_size)).astype(np.uint32)) 
reference_psf_shm = dao.shm('/tmp/reference_psf.im.shm', np.zeros((img_size, img_size)).astype(np.uint32)) 
reference_image_shm = dao.shm('/tmp/reference_image.im.shm' , np.zeros((img_size, img_size)).astype(np.uint32)) 
reference_image_slopes_shm = dao.shm('/tmp/reference_image_slopes.im.shm' , np.zeros((img_size, img_size)).astype(np.float32)) 
