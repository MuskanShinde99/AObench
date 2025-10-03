from astropy.io import fits
#from src.config import config
import os
from matplotlib import pyplot as plt
#from src.shm_loader import shm
import numpy as np

# #folder_calib = config.folder_calib
# folder = '/home/daouser/RISTRETTO/AObench/outputs/Calibration_files_papyrus'
# #folder = '/home/daouser/RISTRETTO/ARPOGE/data'

# mask = fits.getdata(os.path.join(folder, f'mask_3s_pyr.fits'))

# summed = fits.getdata(os.path.join(folder, f'summed_pyr_images_3s_pyr.fits'))

# ref = fits.getdata(os.path.join(folder, f'reference_image_raw.fits'))

# plt.figure()
# plt.imshow(np.log10(summed))
# plt.colorbar()
# plt.title('')

# plt.figure()
# plt.imshow((mask))
# plt.colorbar()
# plt.title('')

# plt.figure()
# plt.imshow(np.log10(mask*summed))
# plt.colorbar()
# plt.title('mask*summed')

# plt.figure()
# plt.imshow(np.log10(mask*ref))
# plt.colorbar()
# plt.title('mask*ref')

# plt.show()


folder = '/home/daouser/RISTRETTO/ARPOGE/data'

#2D plot
img = fits.getdata(os.path.join(folder, f'mask.fits'))

print(img.shape)
plt.figure()
plt.imshow((img))
plt.colorbar()
plt.title('')
plt.show()

# summed_image = np.std(img, axis=0)
# flux_cutoff = 0.04

# flux_limit_upper = summed_image.max() * flux_cutoff
# mask = summed_image >= flux_limit_upper

# print(mask.shape)
# plt.figure()
# plt.imshow(mask)
# plt.colorbar()
# plt.title('mask')

# folder = '/home/daouser/RISTRETTO/AObench/outputs/Calibration_files_papyrus'
# fits.writeto(os.path.join(folder, f'mask_3s_pyr_OnSky.fits'), mask.astype(np.float32), overwrite=True)



# #2D plot
# img = fits.getdata(os.path.join(folder, f'mask_3s_pyr_OnSky.fits'))

# print(img.shape)
# plt.figure()
# plt.imshow((img))
# plt.colorbar()
# plt.title('')
# plt.show()

# # mask = fits.getdata(os.path.join(folder, f'mask.fits'))

# # print(mask.shape)
# # plt.figure()
# # plt.imshow((mask))
# # plt.colorbar()
# # plt.title('')
# # plt.show()






# #1D plot
# # data = fits.getdata(os.path.join(folder, f'2025-09-26_11-40-30/dm3.fits'))

# # print(data.shape)
# # plt.figure()
# # plt.plot(data)
# # plt.title('')
# # plt.show()


# # # plt.show()

# # # U, V, T = np.linalg.svd(img)

# # # plt.figure()
# # # plt.plot(V)
# # # plt.yscale('log')
# # # plt.show()
