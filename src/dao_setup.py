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
from types import SimpleNamespace


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

LASER_PORT = "/dev/ttyUSB0"
LASER_CHANNEL = 1
LASER_CURRENT = 49  # mA, good value for pyramid images


def _init_laser():
    las = Laser(LASER_PORT, LASER_CHANNEL)
    las.set_current(LASER_CURRENT)
    print("Laser is Accessible")
    return las


las = _init_laser()
  
#%% Configuration Camera

def _init_cameras():
    camera_wfs = Camera('/tmp/cam1.im.shm')
    camera_fp = Camera('/tmp/cam2.im.shm')

    fps_wfs = dao.shm('/tmp/cam1Fps.im.shm')
    fps_wfs.set_data(fps_wfs.get_data() * 0 + 300)

    fps_fp = dao.shm('/tmp/cam2Fps.im.shm')
    fps_fp.set_data(fps_fp.get_data() * 0 + 20)

    img_size_wfs_cam = camera_wfs.get_data().shape[0]
    img_size_fp_cam = camera_fp.get_data().shape[0]

    return camera_wfs, camera_fp, img_size_wfs_cam, img_size_fp_cam


camera_wfs, camera_fp, img_size_wfs_cam, img_size_fp_cam = _init_cameras()

#%% Configuration SLM

# Initializes the SLM library

def _init_slm():
    slm = SLM('/tmp/slm.im.shm')
    print('SLM is open')

    dataWidth, dataHeight = slm.get_data().shape[1], slm.get_data().shape[0]
    pixel_size = 8e-3  # Pixel size in mm
    wait_time = 0.15

    return slm, dataWidth, dataHeight, pixel_size, wait_time


slm, dataWidth, dataHeight, pixel_size, wait_time = _init_slm()


#%% Create Pupil grid

pupil_size = 4  # [mm]
blaze_period_outer = 20
blaze_period_inner = 15
tilt_amp_outer = 150
tilt_amp_inner = -70.5  # -70.5 -67 -40


def _create_pupil():
    npix_pupil = int(pupil_size / pixel_size)
    pupil_grid = make_pupil_grid(
        [dataWidth, dataHeight], [dataWidth * pixel_size, dataHeight * pixel_size]
    )
    generator = make_obstructed_circular_aperture(pupil_size, 0, 0, 0)
    pupil_mask = evaluate_supersampled(generator, pupil_grid, 1)
    pupil_mask = pupil_mask.reshape(dataHeight, dataWidth).astype(bool)

    oversizing = 1.1
    npix_small = int(npix_pupil * oversizing)
    small_grid = make_pupil_grid(npix_small, npix_small * pixel_size)

    offset_height = (dataHeight - npix_small) // 2
    offset_width = (dataWidth - npix_small) // 2

    small_mask = pupil_mask[
        offset_height : offset_height + npix_small,
        offset_width : offset_width + npix_small,
    ]

    return (
        pupil_grid,
        pupil_mask,
        small_grid,
        small_mask,
        npix_small,
        offset_height,
        offset_width,
    )


pupil_grid, pupil_mask, small_pupil_grid, small_pupil_mask, npix_small_pupil_grid, offset_height, offset_width = _create_pupil()

#%% Configuration deformable mirror


def _init_dm():
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

    dm_flat = np.zeros(nact ** 2)

    return (
        deformable_mirror,
        dm_modes_full,
        dm_modes,
        nmodes_dm,
        nact_total,
        nact_valid,
        nact,
        dm_flat,
    )


(
    deformable_mirror,
    dm_modes_full,
    dm_modes,
    nmodes_dm,
    nact_total,
    nact_valid,
    nact,
    dm_flat,
) = _init_dm()

#%% Define number of KL and Zernike modes

nmodes_dm = nact_valid
nmodes_KL = nact_valid
nmodes_Znk = nact_valid


#%% Load transformation matrices

# From folder 
KL2Act = fits.getdata(folder_transformation_matrices / f'KL2Act_nkl_{nmodes_KL}_nact_{nact}.fits')
KL2Phs = fits.getdata(folder_transformation_matrices / f'KL2Phs_nkl_{nmodes_KL}_npupil_{npix_small_pupil_grid}.fits')


#%%


def _prepare_dm():
    data_pupil = create_slm_circular_pupil(
        tilt_amp_outer, tilt_amp_inner, pupil_size, pupil_mask, slm
    )

    zernike_basis = make_zernike_basis(11, pupil_size, pupil_grid)
    zernike_basis = [mode / np.ptp(mode) for mode in zernike_basis]
    zernike_basis = np.asarray(zernike_basis)

    data_focus = 0.4 * zernike_basis[3].reshape(dataHeight, dataWidth)
    data_pupil = data_pupil + data_focus

    tt_amplitudes = [-1.662106229803814, 0.09673273462756615]
    tt_amplitude_matrix = np.diag(tt_amplitudes)
    tt_matrix = tt_amplitude_matrix @ KL2Act[0:2, :]
    data_tt = (tt_matrix[0] + tt_matrix[1]).reshape(nact ** 2)

    othermodes_amplitudes = [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    othermodes_amplitude_matrix = np.diag(othermodes_amplitudes)
    othermodes_matrix = othermodes_amplitude_matrix @ KL2Act[2:10, :]
    data_othermodes = np.sum(othermodes_matrix, axis=0)

    dm_flat = data_tt + data_othermodes
    _setup = SimpleNamespace(nact=nact, dm_flat=dm_flat)
    set_dm_actuators(deformable_mirror, dm_flat=dm_flat, setup=_setup)

    data_pupil_outer = np.copy(data_pupil)
    data_pupil_outer[pupil_mask] = 0

    data_pupil_inner = np.copy(
        data_pupil[
            offset_height : offset_height + npix_small_pupil_grid,
            offset_width : offset_width + npix_small_pupil_grid,
        ]
    )
    data_pupil_inner[~small_pupil_mask] = 0

    return (
        data_pupil,
        data_focus,
        tt_amplitudes,
        othermodes_amplitudes,
        dm_flat,
        data_pupil_outer,
        data_pupil_inner,
    )


(
    data_pupil,
    data_focus,
    tt_amplitudes,
    othermodes_amplitudes,
    dm_flat,
    data_pupil_outer,
    data_pupil_inner,
) = _prepare_dm()


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
        self.nact = nact
        self.data_pupil_outer = data_pupil_outer
        self.data_pupil_inner = data_pupil_inner
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

        # Compute the actuator pattern but do not apply it to the DM here.
        # Instead store it in ``dm_flat`` so it can be applied later via
        # :func:`set_data_dm`.
        actuators = data_tt + data_othermodes
        self.actuators = actuators

        # Update the flat map in place so that objects sharing the array (e.g.
        # :class:`DAOSetup`) see the changes.
        if self.dm_flat.shape == actuators.shape:
            self.dm_flat[:] = actuators
        else:
            self.dm_flat = np.asarray(actuators)

        # ``data_dm`` is reset to zero because the DM is not physically updated
        # at this stage. ``set_data_dm`` will generate the actual DM phase when
        # requested.
        self.data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)

        self.data_pupil_outer = np.copy(self.data_pupil)
        self.data_pupil_outer[self.pupil_mask] = 0

        self.data_pupil_inner = np.copy(
            self.data_pupil[offset_height:offset_height + npix_small_pupil_grid,
                             offset_width:offset_width + npix_small_pupil_grid])
        self.data_pupil_inner[~self.small_pupil_mask] = 0


    def update_pupil(self, tt_amplitudes=None, othermodes_amplitudes=None,
                     tilt_amp_outer=None, tilt_amp_inner=None):
        """Update pupil parameters and recompute the DM flat map."""
        if tt_amplitudes is not None:
            self.tt_amplitudes = list(tt_amplitudes)

        if othermodes_amplitudes is not None:
            self.othermodes_amplitudes = list(othermodes_amplitudes)

        if tilt_amp_outer is not None:
            self.tilt_amp_outer = tilt_amp_outer

        if tilt_amp_inner is not None:
            self.tilt_amp_inner = tilt_amp_inner

        self.data_pupil = create_slm_circular_pupil(
            self.tilt_amp_outer, self.tilt_amp_inner, pupil_size, self.pupil_mask, slm
        )
        self.data_pupil = self.data_pupil + self.data_focus
        self._recompute_dm()
        # Return the updated flat map so callers can apply it if needed
        return self.dm_flat


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


