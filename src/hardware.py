import dao
from DEVICES_3.Thorlabs.MCLS1 import mcls1
from hcipy import (
    DeformableMirror as HCIPyDeformableMirror,
    ModeBasis,
    make_gaussian_influence_functions,
)
from skimage.transform import resize
import numpy as np

class ShmWrapper:
    """Simple wrapper exposing the underlying dao.shm object."""
    def __init__(self, shm_path):
        self.shm = dao.shm(shm_path)

    def __getattr__(self, attr):
        return getattr(self.shm, attr)

class Camera(ShmWrapper):
    pass

class SLM(ShmWrapper):
    pass

class Laser:
    """Thin wrapper around the Thorlabs MCLS1 laser."""
    def __init__(self, port="/dev/ttyUSB0", channel=1):
        self.dev = mcls1(port)
        self.dev.set_channel(channel)

    def __getattr__(self, attr):
        return getattr(self.dev, attr)


class DM:
    """Fake deformable mirror hardware using HCIPy's implementation."""

    def __init__(self, pupil_grid, pupil_mask, pupil_size, nact, crosstalk=0.3):
        self.pupil_grid = pupil_grid
        self.pupil_mask = pupil_mask
        self.pupil_size = pupil_size
        self.nact = nact

        self.dm_modes_full = make_gaussian_influence_functions(
            pupil_grid, nact, pupil_size / (nact - 1), crosstalk=crosstalk
        )

        self.valid_actuators_mask = resize(
            np.logical_not(pupil_mask), (nact, nact), order=0,
            anti_aliasing=False, preserve_range=True
        ).astype(int)
        self.valid_actuator_indices = np.column_stack(
            np.where(self.valid_actuators_mask)
        )
        self.nact_total = self.valid_actuators_mask.size
        self.nact_outside = np.sum(self.valid_actuators_mask)
        self.nact_valid = self.nact_total - self.nact_outside

        dm_modes = np.asarray(self.dm_modes_full)
        for x, y in self.valid_actuator_indices:
            dm_modes[x * nact + y] = 0
        self.dm_modes = ModeBasis(dm_modes.T, pupil_grid)

        self.mirror = HCIPyDeformableMirror(self.dm_modes_full)
        self.nmodes_dm = self.mirror.num_actuators
        self.mirror.flatten()

    def __getattr__(self, attr):
        return getattr(self.mirror, attr)


# Backwards compatibility: export HCIPy's DeformableMirror
DeformableMirror = HCIPyDeformableMirror



