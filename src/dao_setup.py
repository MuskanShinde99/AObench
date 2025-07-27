#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module initializing AO bench hardware and pupil configuration."""

# Import Libraries
from hcipy import (
    evaluate_supersampled,
    make_obstructed_circular_aperture,
    make_pupil_grid,
    make_zernike_basis,
)
import numpy as np
from astropy.io import fits
from dataclasses import dataclass


# Import Specific Modules
from src.config import config
from src.hardware import Camera, SLM, Laser, DM
import dao
from src.utils import compute_data_slm, set_default_setup, set_dm_actuators
from src.circular_pupil_functions import create_slm_circular_pupil

ROOT_DIR = config.root_dir
folder_calib = config.folder_calib
folder_pyr_mask = config.folder_pyr_mask
folder_transformation_matrices = config.folder_transformation_matrices
folder_closed_loop_tests = config.folder_closed_loop_tests
folder_turbulence = config.folder_turbulence
folder_gui = config.folder_gui

#%% Start the laser

channel = 1
las = Laser("/dev/ttyUSB0", channel)
# las.enable(1)  # 1 to turn on laser, 0 to turn off
las.set_current(49)  # 55mA is a good value for pyramid images
print('Laser is Accessible')
  
#%% Configuration Camera

# To set camera
camera_wfs = Camera('/tmp/cam1.im.shm')
camera_fp = Camera('/tmp/cam2.im.shm')

fps_wfs = dao.shm('/tmp/cam1Fps.im.shm')
fps_wfs.set_data(fps_wfs.get_data()*0+300)

fps_fp = dao.shm('/tmp/cam2Fps.im.shm')
fps_fp.set_data(fps_fp.get_data()*0+20)

img = camera_wfs.get_data()
img_size_wfs_cam = img.shape[0]

img_fp = camera_fp.get_data()
img_size_fp_cam = img_fp.shape[0]

# # To get camera image
# camera_wfs.get_data()
# camera_fp.get_data()

#%% Configuration SLM

# Initializes the SLM library
slm = SLM('/tmp/slm.im.shm')
print('SLM is open')

# Get SLM dimensions
dataWidth, dataHeight = slm.get_data().shape[1], slm.get_data().shape[0]
pixel_size = 8e-3  # Pixel size in mm

wait_time = 0.15

# dataWidth = slm.width_px
# dataHeight = slm.height_px
# pixel_size = slm.pixelsize_m * 1e3  # Pixel size in mm


#%% Create Pupil grid

# Parameters for the circular pupil
pupil_size = 4  # [mm]
npix_pupil = int(pupil_size / pixel_size)   # Convert pupil size to pixels
blaze_period_outer = 20
blaze_period_inner = 15
tilt_amp_outer = 150
tilt_amp_inner = -70.5  # -70.5 -67 -40

# Create the circular pupil mask
pupil_grid = make_pupil_grid([dataWidth, dataHeight], [dataWidth * pixel_size, dataHeight * pixel_size])
vlt_aperture_generator = make_obstructed_circular_aperture(pupil_size, 0, 0, 0)
pupil_mask = evaluate_supersampled(vlt_aperture_generator, pupil_grid, 1)
pupil_mask =pupil_mask.reshape(dataHeight, dataWidth)
pupil_mask = pupil_mask.astype(bool)

#%% Create a pupil grid for a smaller pupil area

# Set up pupil grid dimensions with size 1.1 times the pupil size
oversizing = 1.1
npix_small_pupil_grid = int(npix_pupil * oversizing) 
small_pupil_grid = make_pupil_grid(npix_small_pupil_grid, npix_small_pupil_grid * pixel_size)
# print('New  small pupil grid created')
# print('Pupil grid shape:', npix_small_pupil_grid, npix_small_pupil_grid)

# Calculate offsets to center the pupil grid with respect to the SLM grid
offset_height = (dataHeight - npix_small_pupil_grid) // 2
offset_width = (dataWidth - npix_small_pupil_grid) // 2

# Create a grid mask 
small_pupil_grid_mask = np.zeros((dataHeight, dataWidth), dtype=bool)
small_pupil_grid_mask[offset_height:offset_height + npix_small_pupil_grid, offset_width:offset_width + npix_small_pupil_grid] = 1

# Create the circular pupil mask for small square grid
small_pupil_mask = pupil_mask[offset_height:offset_height + npix_small_pupil_grid, 
                               offset_width:offset_width + npix_small_pupil_grid]
# plt.figure()
# plt.imshow(small_pupil_mask)
# plt.colorbar()
# plt.title('Small Pupil Mask')
# plt.show()

#%% Configuration deformable mirror

# Number of actuators
nact = 17

deformable_mirror = DM(
    small_pupil_grid,
    small_pupil_mask,
    pupil_size,
    nact,
)

dm_modes_full = deformable_mirror.dm_modes_full
dm_modes = deformable_mirror.dm_modes
nmodes_dm = deformable_mirror.nmodes_dm
nact_total = deformable_mirror.nact_total
nact_valid = deformable_mirror.nact_valid

dm_flat = np.zeros(nact**2)

# Flatten the DM surface and set actuator values
# deformable_mirror.flatten()
# deformable_mirror.actuators = np.ones(nact**2)
# plt.figure()
# plt.imshow(deformable_mirror.opd.shaped)
# plt.colorbar()
# plt.title('Deformable Mirror Surface OPD')
# plt.show()

#%% Define number of KL and Zernike modes

nmodes_dm = nact_valid
nmodes_KL = nact_valid
nmodes_Znk = nact_valid


#%% Load transformation matrices

# From folder 
KL2Act = fits.getdata(folder_transformation_matrices / f'KL2Act_nkl_{nmodes_KL}_nact_{nact}.fits')
KL2Phs = fits.getdata(folder_transformation_matrices / f'KL2Phs_nkl_{nmodes_KL}_npupil_{npix_small_pupil_grid}.fits')


#%%
# Create circular pupil
data_pupil = create_slm_circular_pupil(tilt_amp_outer, tilt_amp_inner, pupil_size, pupil_mask, slm)

# Create Zernike basis
zernike_basis = make_zernike_basis(11, pupil_size, pupil_grid)
zernike_basis = [mode / np.ptp(mode) for mode in zernike_basis]
zernike_basis = np.asarray(zernike_basis)

data_focus = 0.4*zernike_basis[3].reshape(dataHeight, dataWidth)

data_pupil = data_pupil + data_focus

# [-1.6510890005150187, 0.14406016044318903]
# Create a Tip-Tilt (TT) matrix with specified amplitudes as the diagonal elements
tt_amplitudes = [-1.6643032179340225, 0.09163513242902432] # Tip and Tilt amplitudes
tt_amplitude_matrix = np.diag(tt_amplitudes)
tt_matrix = tt_amplitude_matrix @ KL2Act[0:2, :]  # Select modes 1 (tip) and 2 (tilt)

data_tt = (tt_matrix[0] + tt_matrix[1]).reshape(nact**2)

othermodes_amplitudes = [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Focus (mode 3) + modes 4 to 10
othermodes_amplitude_matrix = np.diag(othermodes_amplitudes)
othermodes_matrix = othermodes_amplitude_matrix @ KL2Act[2:10, :]  # Select modes 3 (focus) to 10

data_othermodes = np.sum(othermodes_matrix, axis=0)

#Put the modes on the dm
data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
deformable_mirror.flatten()
deformable_mirror.actuators = data_tt + data_othermodes  # Add TT and higher-order terms to pupil
#set_dm_actuators(deformable_mirror, data_tt + data_othermodes, pupil_setup)
data_dm[:, :] = deformable_mirror.opd.shaped/2 #divide by 2 is very important to get the proper phase. because for this phase to be applied the slm surface needs to half of it.

# Combine the DM surface with the pupil
# ``data_dm`` is defined on the small pupil grid while ``data_pupil`` has the
# full SLM dimensions.  To create a meaningful pattern for the SLM we first
# split ``data_pupil`` into an outer (full size) and inner (small pupil sized)
# part and then add ``data_dm`` to the inner part.  The result is wrapped and
# inserted back into the full pupil.

# Create a new `data_pupil_outer` with the same size as `data_pupil`
data_pupil_outer = np.copy(data_pupil)
data_pupil_outer[pupil_mask] = 0  # Zero out inner region given by `pupil_mask`

# Create a new `data_pupil_inner` with the same size as the small pupil mask
data_pupil_inner = np.copy(
    data_pupil[offset_height:offset_height + npix_small_pupil_grid,
               offset_width:offset_width + npix_small_pupil_grid])
data_pupil_inner[~small_pupil_mask] = 0  # Zero out region outside the mask

# Wrap and insert DM data into the pupil
data_pupil_inner_new = data_pupil_inner + data_dm


class PupilSetup:
    """Encapsulate pupil parameters and provide update utilities."""

    def __init__(self):
        # Register this instance as the default setup so utility functions
        # can rely on it without explicitly passing it around.
        set_default_setup(self)
        self.tilt_amp_outer = tilt_amp_outer
        self.tilt_amp_inner = tilt_amp_inner
        self.tt_amplitudes = list(tt_amplitudes)
        self.othermodes_amplitudes = list(othermodes_amplitudes)
        self.data_pupil = data_pupil
        self.data_focus = data_focus
        self.data_dm = data_dm
        self.nact = nact
        self.data_pupil_outer = data_pupil_outer
        self.data_pupil_inner = data_pupil_inner
        self.data_pupil_inner_new = data_pupil_inner_new
        self.actuators = np.zeros(nact**2)
        # Store masks for later use when recomputing the pupil
        self.pupil_mask = pupil_mask
        self.small_pupil_mask = small_pupil_mask
        self.dm_flat = dm_flat
        self.data_slm = compute_data_slm()

    def _recompute_dm(self):
        """(Re)compute DM contribution and assemble the pupil."""
        tt_matrix = np.diag(self.tt_amplitudes) @ KL2Act[0:2, :]
        data_tt = (tt_matrix[0] + tt_matrix[1])

        othermodes_matrix = np.diag(self.othermodes_amplitudes) @ KL2Act[2:10, :]
        data_othermodes = np.sum(othermodes_matrix, axis=0)

        self.data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
        deformable_mirror.flatten()
        # deformable_mirror.actuators = data_tt + data_othermodes  # Add TT and higher-order terms to pupil
        actuators = data_tt + data_othermodes
        set_dm_actuators(deformable_mirror, actuators)
        self.actuators = actuators
        self.data_dm[:, :] = deformable_mirror.opd.shaped / 2

        self.data_pupil_outer = np.copy(self.data_pupil)
        self.data_pupil_outer[self.pupil_mask] = 0

        self.data_pupil_inner = np.copy(
            self.data_pupil[offset_height:offset_height + npix_small_pupil_grid,
                             offset_width:offset_width + npix_small_pupil_grid])
        self.data_pupil_inner[~self.small_pupil_mask] = 0

        self.data_pupil_inner_new = self.data_pupil_inner + self.data_dm


    def update_pupil(self, new_tt_amplitudes=None, new_othermodes_amplitudes=None,
                     new_tilt_amp_outer=None, new_tilt_amp_inner=None):
        if new_tt_amplitudes is not None:
            self.tt_amplitudes = list(new_tt_amplitudes)

        if new_othermodes_amplitudes is not None:
            self.othermodes_amplitudes = list(new_othermodes_amplitudes)

        if new_tilt_amp_outer is not None:
            self.tilt_amp_outer = new_tilt_amp_outer

        if new_tilt_amp_inner is not None:
            self.tilt_amp_inner = new_tilt_amp_inner

        self.data_pupil = create_slm_circular_pupil(
            self.tilt_amp_outer, self.tilt_amp_inner, pupil_size, self.pupil_mask, slm
        )
        self.data_pupil = self.data_pupil + self.data_focus
        self._recompute_dm()
        return self.actuators


pupil_setup = PupilSetup()

def update_pupil(*args, **kwargs):
    """Wrapper for backward compatibility."""
    return pupil_setup.update_pupil(*args, **kwargs)


@dataclass
class DAOSetup:
    """Bundled access to frequently used setup objects."""

    las: Laser
    camera_wfs: Camera
    camera_fp: Camera
    slm: SLM
    deformable_mirror: DM
    pupil_setup: PupilSetup
    wait_time: float
    dataWidth: int
    dataHeight: int
    npix_small_pupil_grid: int
    pupil_size: float
    pixel_size: float
    folder_calib: str
    folder_pyr_mask: str
    folder_transformation_matrices: str
    folder_closed_loop_tests: str
    folder_turbulence: str
    folder_gui: str
    img_size_wfs_cam: int
    img_size_fp_cam: int
    nact: int
    nact_valid: int
    nact_total: int
    nmodes_dm: int
    nmodes_KL: int
    nmodes_Znk: int
    small_pupil_mask: np.ndarray
    pupil_mask: np.ndarray
    dm_flat: np.ndarray


def init_setup() -> DAOSetup:
    """Return a :class:`DAOSetup` instance with initialized components."""

    return DAOSetup(
        las=las,
        camera_wfs=camera_wfs,
        camera_fp=camera_fp,
        slm=slm,
        deformable_mirror=deformable_mirror,
        dm_flat=dm_flat,
        pupil_setup=pupil_setup,
        wait_time=wait_time,
        dataWidth=dataWidth,
        dataHeight=dataHeight,
        npix_small_pupil_grid=npix_small_pupil_grid,
        pupil_size=pupil_size,
        pixel_size=pixel_size,
        folder_calib=str(folder_calib),
        folder_pyr_mask=str(folder_pyr_mask),
        folder_transformation_matrices=str(folder_transformation_matrices),
        folder_closed_loop_tests=str(folder_closed_loop_tests),
        folder_turbulence=str(folder_turbulence),
        folder_gui=str(folder_gui),
        img_size_wfs_cam=img_size_wfs_cam,
        img_size_fp_cam=img_size_fp_cam,
        nact=nact,
        nact_valid=nact_valid,
        nact_total=nact_total,
        nmodes_dm=nmodes_dm,
        nmodes_KL=nmodes_KL,
        nmodes_Znk=nmodes_Znk,
        small_pupil_mask=small_pupil_mask,
        pupil_mask=pupil_mask,
    )


