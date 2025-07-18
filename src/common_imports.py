"""Common imports for AO bench scripts.

This module centralizes frequently used imports and provides a ready-to-use
``setup`` object. Scripts under ``scripts_with_dao`` can simply do::

    from src.common_imports import *

This keeps each script short and ensures consistent initialization.
"""

# Standard library
import os
import sys
import time
from pathlib import Path

# Third-party libraries
import numpy as np
import scipy
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from PIL import Image
from tqdm import tqdm
from pypylon import pylon

# AO bench modules
from src.config import config
from src.dao_setup import init_setup, las
from src.utils import *
from src.circular_pupil_functions import *
from src.flux_filtering_mask_functions import *
from src.tilt_functions import *
from src.calibration_functions import *
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil
from src.transformation_matrices_functions import *
from src.psf_centring_algorithm_functions import *
from src.create_shared_memories import *
from src.scan_modes_functions import *
from src.ao_loop_functions import *

# Initialize the default setup for convenience
setup = init_setup()

# Export all loaded names except for private ones
__all__ = [name for name in globals() if not name.startswith('_')]
