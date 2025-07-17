import dao
from DEVICES_3.Thorlabs.MCLS1 import mcls1
from hcipy import DeformableMirror as _HCIPY_DM
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


class DeformableMirror(_HCIPY_DM):
    """Deformable mirror hardware backed by shared memory."""

    def __init__(self, influence_functions, shm_path="/tmp/dm_act.im.shm"):
        super().__init__(influence_functions)
        self.shm = dao.shm(shm_path, np.zeros(self.num_actuators, dtype=np.float32))
        try:
            init_val = self.shm.get_data()
            if init_val.size == self.num_actuators:
                _HCIPY_DM.actuators.fset(self, init_val.ravel())
        except Exception:
            self.shm.set_data(self.actuators.astype(np.float32))

    @property  # type: ignore[override]
    def actuators(self):
        return _HCIPY_DM.actuators.fget(self)

    @actuators.setter
    def actuators(self, values):
        _HCIPY_DM.actuators.fset(self, values)
        self.shm.set_data(np.asarray(self._actuators, dtype=np.float32))
