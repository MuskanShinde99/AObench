# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:05:59 2024

@author: RISTRETTO
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import os

# Load FITS file
folder = os.path.join('C:\\', 'Users', 'RISTRETTO', 'RISTRETTO_AO_bench_images')
filename = 'processed_response_cube_zernike_push-pull_pup_4mm_nact_11_amp_0.1_3s_pyr.fits'
file_path = os.path.join(folder, filename)
with fits.open(file_path) as hdul:
    data_cube = hdul[0].data  # Assuming the data is in the primary HDU

# Check dimensions (time or Z dimension is typically first)
print(f"Data shape: {data_cube.shape}")

# Create a list to store images
images = []

# Loop through each frame in the data cube
for frame in data_cube:

    # Use matplotlib to render the frame as an image
    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='gray', origin='lower', vmax=3e-5, vmin=-3e-5)
    ax.axis('off')
    
    # Add a colorbar to the figure
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')

    # Save to a temporary buffer and load into PIL
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
    plt.close(fig)

    # Append the PIL image to the list
    images.append(image)

# Save the images as a GIF
output_gif = "processed_response_cube_zernike_push-pull_pup_4mm_nact_11_amp_0.1_3s_pyr.gif"  # Replace with your desired output name
images[0].save(os.path.join(folder, output_gif), save_all=True, append_images=images[1:], loop=0, duration=100)

print(f"GIF saved as {output_gif}")