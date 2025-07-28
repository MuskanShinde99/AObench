#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:56:36 2025

@author: ristretto-dao
"""

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
import time
from astropy.io import fits
from dataclasses import dataclass
from types import SimpleNamespace

# Import Specific Modules
from src.config import config
from src.hardware import Camera
import dao
from src.utils import compute_data_slm, set_default_setup, DEFAULT_SETUP
from src.shm_loader import shm

ROOT_DIR = config.root_dir
folder_calib = config.folder_calib
folder_pyr_mask = config.folder_pyr_mask
folder_transformation_matrices = config.folder_transformation_matrices
folder_closed_loop_tests = config.folder_closed_loop_tests
folder_turbulence = config.folder_turbulence
folder_gui = config.folder_gui

def set_dm_actuators(actuators=None, dm_flat=None, setup=None, **kwargs):
    """Set DM actuators and update the shared memory grid.

    If a deformable mirror instance is available it is updated as well.  When
    running on a minimal setup (no DM hardware), the actuator pattern is simply
    written to the shared memory segment.
    """

    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP

    if actuators is None:
        actuators = np.zeros(setup.nact ** 2)
    if dm_flat is None:
        dm_flat = setup.dm_flat

    actuators = np.asarray(actuators)

    deformable_mirror = kwargs.get(
        "deformable_mirror", getattr(setup, "deformable_mirror", None)
    )

    if deformable_mirror is not None:
        deformable_mirror.actuators = actuators + dm_flat
        actuators_to_store = deformable_mirror.actuators
    else:
        actuators_to_store = actuators + dm_flat

    dm_act_shm = shm.dm_act_shm
    dm_act_shm.set_data(
        np.asarray(actuators_to_store).astype(np.float64).reshape(
            setup.nact, setup.nact
        )
    )


def set_data_dm(actuators=None, *, setup=None, dm_flat=None, **kwargs):
    """Flatten the DM, optionally apply ``actuators`` and, if available, update the SLM."""

    if setup is None:
        if DEFAULT_SETUP is None:
            raise ValueError("No setup provided and no default registered.")
        setup = DEFAULT_SETUP

    slm = kwargs.get("slm", getattr(setup, "slm", None))
    deformable_mirror = kwargs.get(
        "deformable_mirror", getattr(setup, "deformable_mirror", None)
    )
    npix_small_pupil_grid = kwargs.get(
        "npix_small_pupil_grid", getattr(setup, "npix_small_pupil_grid", None)
    )
    wait_time = kwargs.get("wait_time", getattr(setup, "wait_time", 0))
    pupil_setup = kwargs.get("pupil_setup", getattr(setup, "pupil_setup", None))

    if deformable_mirror is not None:
        deformable_mirror.flatten()

    set_dm_actuators(
        actuators, dm_flat=dm_flat, setup=setup, deformable_mirror=deformable_mirror
    )

    if deformable_mirror is not None and npix_small_pupil_grid is not None:
        data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)
        data_dm[:, :] = deformable_mirror.opd.shaped / 2
    else:
        data_dm = np.zeros((setup.nact, setup.nact), dtype=np.float32)

    if slm is not None and pupil_setup is not None:
        data_slm = compute_data_slm(data_dm=data_dm, setup=pupil_setup)
        slm.set_data(data_slm)
        time.sleep(wait_time)
        return actuators, data_dm, data_slm

    return actuators, data_dm, None

#%% Configuration Camera

# To set camera
camera_wfs = Camera('/tmp/cam1.im.shm') #change to CRED3
camera_fp = Camera('/tmp/cam2.im.shm') #change to CBlue

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


#%% Configuration deformable mirror

# Number of actuators
nact = 17

nact_total = nact**2
nact_valid = nact_total #195

dm_flat = np.zeros(nact**2)

#%% Define number of KL and Zernike modes

nmodes_dm = nact_valid
nmodes_KL = 185
nmodes_Znk = nact_valid


#%% Load transformation matrices

# From folder 
KL2Act = fits.getdata(folder_transformation_matrices / f'KL2Act_nkl_{nmodes_KL}_nact_{nact}.fits')


#%%

# [-1.6510890005150187, 0.14406016044318903]
# Create a Tip-Tilt (TT) matrix with specified amplitudes as the diagonal elements
tt_amplitudes = [-1.647661297087426, 0.10306330000366959] # Tip and Tilt amplitudes
tt_amplitude_matrix = np.diag(tt_amplitudes)
tt_matrix = tt_amplitude_matrix @ KL2Act[0:2, :]  # Select modes 1 (tip) and 2 (tilt)

data_tt = (tt_matrix[0] + tt_matrix[1]).reshape(nact**2)

othermodes_amplitudes = [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Focus (mode 3) + modes 4 to 10
othermodes_amplitude_matrix = np.diag(othermodes_amplitudes)
othermodes_matrix = othermodes_amplitude_matrix @ KL2Act[2:10, :]  # Select modes 3 (focus) to 10

data_othermodes = np.sum(othermodes_matrix, axis=0)

#Put the modes on the dm
dm_flat = data_tt + data_othermodes
_setup = SimpleNamespace(
    nact=nact,
    dm_flat=dm_flat,
)
set_dm_actuators(dm_flat=dm_flat, setup=_setup)

# Combine the DM surface with the pupil
# ``data_dm`` is defined on the small pupil grid while ``data_pupil`` has the
# full SLM dimensions.  To create a meaningful pattern for the SLM we first
# split ``data_pupil`` into an outer (full size) and inner (small pupil sized)
# part and then add ``data_dm`` to the inner part.  The result is wrapped and
# inserted back into the full pupil.


class PupilSetup:
    """Encapsulate pupil parameters and provide update utilities."""

    def __init__(self):
        # Register this instance as the default setup so utility functions
        # can rely on it without explicitly passing it around.
        set_default_setup(self)

        self.tt_amplitudes = list(tt_amplitudes)
        self.othermodes_amplitudes = list(othermodes_amplitudes)
        self.nact = nact
        self.actuators = np.zeros(nact**2)
        # Store masks for later use when recomputing the pupil
        self.dm_flat = dm_flat

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


    def update_pupil(self, tt_amplitudes=None, othermodes_amplitudes=None,):
        """Update pupil parameters and recompute the DM flat map."""
        if tt_amplitudes is not None:
            self.tt_amplitudes = list(tt_amplitudes)

        if othermodes_amplitudes is not None:
            self.othermodes_amplitudes = list(othermodes_amplitudes)


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

    camera_wfs: Camera
    camera_fp: Camera
    pupil_setup: PupilSetup
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
    dm_flat: np.ndarray


def init_setup() -> DAOSetup:
    """Return a :class:`DAOSetup` instance with initialized components."""

    return DAOSetup(
        camera_wfs=camera_wfs,
        camera_fp=camera_fp,
        pupil_setup=pupil_setup,
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
        dm_flat=dm_flat,
    )


