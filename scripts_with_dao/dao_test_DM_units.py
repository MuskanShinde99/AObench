import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Import Specific Modules
import dao
from src.dao_setup import *  # Import all variables from setup
from src.utils import *

# Flatten the DM surface and set actuator values
deformable_mirror.flatten()

deformable_mirror.flatten()
desired_opd = 1e-9 # 100 nm OPD
deformable_mirror.actuators = desired_opd * np.ones(nact**2)/ 2

plt.figure()
plt.imshow(deformable_mirror.opd.shaped)
plt.colorbar()
plt.title('Deformable Mirror Surface OPD')
plt.show()