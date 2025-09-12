from astropy.io import fits
from src.config import config
import os
from matplotlib import pyplot as plt
#from src.shm_loader import shm
import numpy as np

folder_calib = config.folder_calib

#img = fits.getdata(os.path.join(folder_calib, f'summed_pyr_images_3s_pyr.fits'))
img = fits.getdata('/home/daouser/RISTRETTO/AObench/src/record/2025-08-08_18-29-42/cblue.fits')
# bias_image_shm = shm.bias_image_shm
# bias_image_shm.set_data(bias.astype(np.float64))
print(img.shape)
plt.figure()
#plt.imshow(np.log10(np.mean(img[0])/np.mean(img[0]).max()))
plt.imshow((img[0]/img[0].max()))
plt.colorbar()
plt.clim([0.65,0.75])
plt.title('psf')
plt.show()


# plt.figure()
# plt.imshow(img[1])
# plt.title('mode 1')
# plt.colorbar()
# plt.show()

# # plt.figure()
# # plt.imshow(img[31])
# # plt.title('mode 31')
# # plt.colorbar()

# # plt.show()

# # U, V, T = np.linalg.svd(img)

# # plt.figure()
# # plt.plot(V)
# # plt.yscale('log')
# # plt.show()
