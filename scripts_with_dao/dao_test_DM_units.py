import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from src.config import config

ROOT_DIR = config.root_dir

# Import Specific Modules
import dao
from src.dao_setup import init_setup
setup = init_setup()  # Import all variables from setup
from src.utils import *
from src.dao_setup import set_dm_actuators

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()
desired_opd = 1e-9 # 100 nm OPD
act_pos = desired_opd * np.ones(nact**2)/ 2
set_dm_actuators(
    act_pos,
    setup=setup,
)

plt.figure()
plt.imshow(deformable_mirror.opd.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface OPD')
plt.show()
