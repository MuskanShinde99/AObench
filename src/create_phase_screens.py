# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 15:22:51 2024

@author: RISTRETTO
"""

from matplotlib import pyplot as plt
from hcipy import *
import numpy as np
import os
import sys
from pathlib import Path
from astropy.io import fits
import time

#%%

#Create a circular pupil on SLM
blaze_period_outer = 5
blaze_period_inner = 10
pupil_size = 4  # [mm]
dataWidth =  1920 # slm width
dataHeight =  1200 # slm height
pixel_size = 7.999999979801942e-06*1e3  #  slm pixel size in mm

npix_pupil = int(pupil_size/pixel_size)
grid_oversizing = 1.1
pupil_grid_width = int(npix_pupil*grid_oversizing)
pupil_grid_height = int(npix_pupil*grid_oversizing)

# Create the circular pupil mask fr creating kl basis:
x = np.linspace(-pupil_grid_width / 2, pupil_grid_width / 2, pupil_grid_width)
y = np.linspace(-pupil_grid_height / 2, pupil_grid_height / 2, pupil_grid_height)
xx, yy = np.meshgrid(x, y)
rr = np.abs(xx + 1j * yy)
small_pupil_mask = rr < npix_pupil / 2# 

vlt_aperture = small_pupil_mask.flatten()

plt.figure()
plt.imshow(small_pupil_mask)
plt.colorbar()
plt.title('VLT aperture')
plt.show()

#%%
turbulence_pupil_size = 1.52 #m
turbulence_grid = make_pupil_grid(int(npix_pupil*grid_oversizing), turbulence_pupil_size)

#Turbulence parameters
seeing = 2 # arcsec @ 500nm (convention)
outer_scale = 40 # meter
tau0 = 0.005 #ms
#velocity = 5  #m/s
wl = 1700e-9

fried_parameter = seeing_to_fried_parameter(seeing)
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
velocity = 0.314 * fried_parameter / tau0
#tau0 = (0.314 * fried_parameter) / velocity  # Compute tau0 based on velocity

print('r0   = {0:.1f}cm'.format(fried_parameter * 100))
print('L0   = {0:.1f}m'.format(outer_scale))
print('tau0 = {0:.1f}ms'.format(tau0 * 1000))
print('v    = {0:.1f}m/s'.format(velocity))

start_time = time.time()
layer = InfiniteAtmosphericLayer(turbulence_grid, Cn_squared, outer_scale, velocity)
end_time = time.time()
print('Turbulence layer created')
print(f'Time to create turbulence: {end_time - start_time} s')
# Time to create turbulence: 5.780646324157715 s

phase_screen_phase = layer.phase_for(wl)# in radian
phase_screen_opd = (phase_screen_phase * (wl / (2 * np.pi)) * 1e9)

plt.figure()
plt.imshow(phase_screen_phase.reshape(pupil_grid_width, pupil_grid_width), cmap='RdBu')
plt.colorbar()
plt.title('Phase')
plt.show()

#%%

from src.config import config

ROOT_DIR = config.root_dir

# Output folders
folder_calib = config.folder_calib
folder_pyr_mask = config.folder_pyr_mask
folder_transformation_matrices = config.folder_transformation_matrices
folder_closed_loop_tests = config.folder_closed_loop_tests
folder_turbulence = config.folder_turbulence


# Define parameters
wavelengths = [500e-9]  # Measurement wavelengths in meters
nframes = 500  # Number of frames
framerates = [1.0]  # AO loop speeds in kHz

# Loop over each framerate
for framerate in framerates:
    dt = 1 / (framerate * 1e3)  # Convert kHz to Hz, then take inverse
    t_end = nframes * dt  # Compute total duration

    print(f"Processing for framerate: {framerate} kHz")

    # Loop over each wavelength
    for wl in wavelengths:
        phase_cube = []

        layer.reset()  # Reset the layer to start from t=0

        start_time = time.time()
        for i in range(nframes):
            t = i * dt
            print(f"Setting layer.t = {t:.6f}. Current layer time = {layer.t}")
            layer.t = t

            phase_screen_phase = layer.phase_for(wl) # in radians
            # phase_screen_opd = (phase_screen_phase * (wl / (2 * np.pi)) * 1e9) # in nm
            
            # Append the 2D phase screen data to the 3D data cube
            phase_cube.append(phase_screen_phase.reshape(pupil_grid_width, pupil_grid_width))

        end_time = time.time()
        print(f'Time to create {nframes} phase screen cube for {wl*1e9:.0f} nm at {framerate} kHz: {end_time - start_time:.2f} s')

        # Convert the list to a 3D numpy array
        phase_cube = np.array(phase_cube)
       
        # Construct output filename
        output_filename = (f'turbulence_cube_phase_seeing_{seeing}arcsec_L_{outer_scale}m_tau0_5ms_'
                           f'lambda_{wl*1e9:.0f}nm_pup_{turbulence_pupil_size}m_{framerate:.1f}kHz_cube3.fits')
        output_file_path = os.path.join(folder_turbulence / 'Papyrus', output_filename)

        # Save the 3D data cube as a FITS file with overwrite enabled
        fits.writeto(output_file_path, phase_cube, overwrite=True)
        print(f"Saved: {output_file_path}")
