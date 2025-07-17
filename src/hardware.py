import dao
from DEVICES_3.Thorlabs.MCLS1 import mcls1
from hcipy import DeformableMirror

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



