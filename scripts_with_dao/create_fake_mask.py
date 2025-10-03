from astropy.io import fits
#from src.config import config
import os
from matplotlib import pyplot as plt
#from src.shm_loader import shm
import numpy as np
import cv2

folder = '/home/daouser/RISTRETTO/AObench/outputs/Calibration_files_papyrus'


mask = fits.getdata(os.path.join(folder, f'mask_3s_pyr.fits'))
#mask[:, 20:30] = 0
mask[65:90, 80:90] = 0

plt.figure()
plt.imshow((mask))
plt.colorbar()
plt.title('')

summed = fits.getdata(os.path.join(folder, f'reference_image_raw.fits'))

# Find connected components in the binary mask
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
pupil_centers = centroids[1:]  # Skip background label
radii = [np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) for i in range(1, num_labels)]
radius = int(round(np.mean(radii)))  # Average radius

print('centers', (pupil_centers))
print('radius', radius)

mask_fake = np.zeros_like(mask)
for x,y in pupil_centers:

    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask_area = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2  # Create a circular mask
    mask_fake =mask_fake+mask_area
 
plt.figure()
plt.imshow((mask_fake*summed))
plt.colorbar()
plt.title('')
plt.show()


fits.writeto(os.path.join(folder, 'mask_fake.fits'), mask_fake, overwrite=True)