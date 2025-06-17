# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:05:07 2024

@author: RISTRETTO
"""

from matplotlib import pyplot as plt
from hcipy import *
from PIL import Image
import numpy as np
import os

# Set the Working Directory
os.chdir('/home/laboptic/Documents/optlab-master/PROJECTS_3/RISTRETTO/Banc AO')

# Import Specific Modules
from src.tilt import *
from src.create_circular_pupil import *


#%%

#number of actuators
nact = 10
pupil_size = dao_setup.pupil_size
pupil_grid = dao_setup.pupil_grid
pixel_size = 8e-3 # pixel size in mm 

#create a DM
dm_modes = make_gaussian_influence_functions(pupil_grid, nact, pupil_size/nact, crosstalk=0.3)
deformable_mirror = DeformableMirror(dm_modes)
nmodes_dm = deformable_mirror.num_actuators
print("number of dm modes =", nmodes_dm)

deformable_mirror.flatten()
#deformable_mirror.actuators = np.ones((nact, nact)).flatten()
#deformable_mirror.actuators[52] = 1
print(deformable_mirror.surface.shaped.shape)
plt.figure()
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('deformable mirror surface')
plt.show()

deformable_mirror.actuators[45] = 0.5

plt.figure()
plt.imshow(deformable_mirror.surface.shaped)
plt.colorbar()
plt.title('deformable mirror surface')
plt.show()

# data_slm = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight)
# data_slm[:, :] = deformable_mirror.surface.shaped
# data = data_with_tilts + data_slm

# #wrap data
# data = data%1
# plt.figure()
# plt.imshow(data)
# plt.title('Data sent to SLM')
# plt.colorbar()
# plt.show()


# # Show data on SLM:
# error = slm.showData(data)
# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)


