# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:28:36 2024

@author: RISTRETTO
"""

import numpy as np
import os
import time
from astropy.io import fits
from matplotlib import pyplot as plt
from hcipy import *

#%% Setup pupil on the SLM

pupil_size = 4  # [mm]
pixel_size = 7.999999979801942e-06 * 1e3  # SLM pixel size in mm
npix_pupil = int(pupil_size / pixel_size)
grid_oversizing = 1.1

# Pupil grid dimensions
pupil_grid_width = int(npix_pupil * grid_oversizing)
pupil_grid_height = int(npix_pupil * grid_oversizing)

# Create pupil grid and aperture
pupil_grid = make_pupil_grid([pupil_grid_width, pupil_grid_height], [
                             pupil_grid_width * pixel_size, pupil_grid_height * pixel_size])
pupil_generator = make_obstructed_circular_aperture(pupil_size, 0, 0, 0)
pupil = evaluate_supersampled(pupil_generator, pupil_grid, 1)

# Visualize the pupil
plt.figure()
plt.imshow(pupil.shaped)
plt.colorbar()
plt.title('VLT Aperture')
plt.show()

#%% Setup turbulence parameters and create layers

# Turbulence parameters
seeing = 0.7  # arcsec @ 500nm
outer_scale = 40  # meters
tau0 = 0.005  # seconds
wl = 1000e-9  # Wavelength in nm
fried_parameter = seeing_to_fried_parameter(seeing)
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9) # At 500 nm by convention
velocity = 0.314 * fried_parameter / tau0

# Display turbulence parameters
print(f'r0   = {fried_parameter * 100:.1f} cm')
print(f'L0   = {outer_scale:.1f} m')
print(f'tau0 = {tau0 * 1000:.1f} ms')
print(f'v    = {velocity:.1f} m/s')

# Create atmospheric layer
start_time = time.time()
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
end_time = time.time()
print(f'Turbulence layer created in {end_time - start_time:.2f} seconds')

# Get phase and OPD at specified wavelength
phase_screen_phase = layer.phase_for(wl) * pupil
phase_screen_opd = (phase_screen_phase * (wl / (2 * np.pi)) * 1e9) * pupil  # in nm

# Visualize phase and OPD
plt.figure()
plt.imshow(phase_screen_phase, cmap='RdBu')
plt.colorbar()
plt.title('Phase')
plt.show()

plt.figure()
plt.imshow(phase_screen_opd, cmap='RdBu')
plt.colorbar()
plt.title('OPD')
plt.show()

# Save phase and OPD as FITS files
folder = os.path.join('C:\\', 'Users', 'RISTRETTO', 'RISTRETTO_AO_bench_images', 'Phase_screens')
phase_filename = f'phase_seeing_{seeing}arcsec_L_{outer_scale}m_tau0_{tau0}s_lambda_{wl*1e9}nm_pup_{turbulence_pupil_size}m.fits'
opd_filename = f'opd_seeing_{seeing}arcsec_L_{outer_scale}m_tau0_{tau0}s_lambda_{wl*1e9}nm_pup_{turbulence_pupil_size}m.fits'

fits.writeto(os.path.join(folder, phase_filename), phase_screen_phase, overwrite=True)
fits.writeto(os.path.join(folder, opd_filename), phase_screen_opd, overwrite=True)

#%% Create phase cube for time evolution

wl = 639e-9  # Measurement wavelength
t_end = 0.1  # Total duration (seconds)
nframes = 100  # Number of frames
framerate = int(nframes / t_end) * 1e-3  # AO loop speed (kHz)

# Initialize phase cube list and reset turbulence layer
phase_cube = []
layer.reset() #Reset the the layer to start at time=0

# Generate phase cube for each frame
start_time = time.time()
for t in np.linspace(0, t_end, nframes):
    print(f"Setting layer.t = {t}. Current layer time = {layer.t}")

    layer.t = t # Set layer above the telescope pupil at a given instant of time
    phase_screen_phase = layer.phase_for(wl) * pupil  # in radians
    phase_cube.append(phase_screen_phase)

end_time = time.time()
print(f'Time to create phase cube: {end_time - start_time:.2f} seconds')

# Convert phase cube to numpy array and save as FITS file
phase_cube = np.array(phase_cube)
phase_cube_filename = f'phase_screen_cube_phase_seeing_{seeing}arcsec_L_{outer_scale}m_tau0_{tau0}s_lambda_{wl*1e9}nm_pup_{turbulence_pupil_size}m_{framerate}kHz.fits'
fits.writeto(os.path.join(folder, phase_cube_filename), phase_cube, overwrite=True)

