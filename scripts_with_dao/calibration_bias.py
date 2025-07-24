#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 07:40:29 2025

@author: ristretto-dao
"""

from src.common_imports import *

# Turn off laser
las.enable(0) 
time.sleep(2)  # Allow some time for laser to turn off

# Capture and average 1000 bias frames
n_frames=1000
bias_image = np.median([camera_wfs.get_data() for i in range(n_frames)], axis=0)
bias_image_shm.set_data(bias_image)

# Plot
plt.figure()
plt.imshow(bias_image, cmap='gray')
plt.title('Bias image')
plt.colorbar()
plt.show()

# Save the Bias Image
fits.writeto(os.path.join(folder_calib, f'binned_bias_image.fits'), np.asarray(bias_image), overwrite=True)
