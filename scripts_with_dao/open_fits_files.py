from astropy.io import fits
from src.config import config
import os
from matplotlib import pyplot as plt
#from src.shm_loader import shm
import numpy as np

folder_calib = config.folder_calib
folder = '/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/ARPOGE/record'

#2D plot
# img = fits.getdata(os.path.join(folder, f'2025-09-23_13-59-55/dm3.fits'))

# print(img.shape)
# plt.figure()
# plt.imshow((img[0]/img[0].max()))
# plt.colorbar()
# plt.title('')
# plt.show()

#1D plot
data = fits.getdata(os.path.join(folder, f'2025-09-26_11-40-30/dm3.fits'))

print(data.shape)
plt.figure()
plt.plot(data)
plt.title('')
plt.show()


# # plt.show()

# # U, V, T = np.linalg.svd(img)

# # plt.figure()
# # plt.plot(V)
# # plt.yscale('log')
# # plt.show()
