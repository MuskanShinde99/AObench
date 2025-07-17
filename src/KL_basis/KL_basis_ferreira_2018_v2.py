#!/usr/bin/env python
# coding: utf-8

# In[2]:


from hcipy import *
from src.hardware import DeformableMirror
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from holoeye import detect_heds_module_path  # Move all the example files into the holoeye folder in Lib/site-packages 
from holoeye import slmdisplaysdk

# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()

# Detect SLMs and open a window on the selected SLM
error = slm.open()


# In[20]:


# The procedure follows Ferreira+ 2018, "Numerical estimation of wavefront error breakdown in adaptive optics"
npupil=60 # number of pixels in pupil grid
nact=10 # number of actuators along one diameter
diam_tel=8  # telescope circular aperture in m
diam_grid=8 # pupil grid diameter in meters 


# In[21]:
#Create a circular pupil on SLM
blaze_period_outer = 5
blaze_period_inner = 10
pupil_size = 2  # [mm]
dataWidth =  slm.width_px
dataHeight =  slm.height_px
pixel_size = slm.pixelsize_m*1e3  #  pixel size in mm
Npix_pupil = pupil_size/pixel_size 


#generating the telescope pupil
pupil_grid = make_pupil_grid(npupil, diam_grid)
vlt_aperture_generator = make_obstructed_circular_aperture(diam_tel, 0, 0, 0)
vlt_aperture = evaluate_supersampled(vlt_aperture_generator, pupil_grid, 1)

#plotting
imshow_field(vlt_aperture, cmap='gray')
plt.xlabel('x position(m)')
plt.ylabel('y position(m)')
plt.colorbar()
plt.show()

print(vlt_aperture.shape)


# In[22]:


dm_modes = make_gaussian_influence_functions(pupil_grid, nact, diam_tel/nact, crosstalk=0.3)

# Multiply each row of the dm_modes matrix by the vlt_aperture
#dm_modes = [dm_modes[i] * vlt_aperture for i in np.arange(0, np.asarray(dm_modes).shape[0], 1)]
#dm_modes = ModeBasis(dm_modes, pupil_grid)

deformable_mirror = DeformableMirror(dm_modes)

nmodes_dm = deformable_mirror.num_actuators
print("number of modes =", nmodes_dm)

deformable_mirror.actuators = np.ones((nact, nact)).flatten()
plt.imshow(deformable_mirror.surface.shaped)

deformable_mirror.flatten()

IF=dm_modes.transformation_matrix.toarray()
IF=IF*vlt_aperture[:,np.newaxis]


# In[27]:


from eigenModes_v2 import computeEigenModes
M2C, KLphiVec = computeEigenModes(IF, vlt_aperture, disp=False, sort=True, includeTT=True)


#Petal modes
"""petalx = (pupil_grid.x>0)*1.
petalx -= petalx.mean()
petaly = (pupil_grid.x>0)*1.
petaly -= petalx.mean()

petal1 = (pupil_grid.x>=0)*(pupil_grid.y>=0)*(vlt_aperture>0)
petal2 = (pupil_grid.x<0)*(pupil_grid.y>=0)*(vlt_aperture>0)
petal3 = (pupil_grid.x>=0)*(pupil_grid.y<0)*(vlt_aperture>0)
petal4 = (pupil_grid.x<0)*(pupil_grid.y<0)*(vlt_aperture>0)

plt.imshow(petal1.reshape(60,60))
plt.colorbar()
plt.show()"""

M2C, KLphiVec = computeEigenModes(IF, vlt_aperture, disp=False, sort=True)


# In[24]:


for i in np.arange(0, 99):
    plt.figure()
    plt.imshow(KLphiVec[:,i].reshape(npupil, npupil), cmap='bwr')
    plt.colorbar()






