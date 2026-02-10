nact_total 289
act_pos shape (289,)
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
File ~/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/scripts_with_dao/dao_calibrations_file.py:99
     96 modulation_amp = 2 # in lamda/D
     97 n_iter=500 # number of iternations for dm random commands
---> 99 mask = create_flux_filtering_mask(method, flux_cutoff, KL2Act[0], KL2Act[1],
    100                                modulation_angles, modulation_amp, n_iter, 
    101                                create_summed_image=True, verbose=False, verbose_plot=True,
    102                                OnSky=False, )
    104 print(f"Mask dimensions: {mask.shape}")
    106 #valid_pixels_mask_shm.set_data(mask)
    107 
    108 # Get valid pixels from the mask
    109 #put with in the function to create mask

File ~/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/src/flux_filtering_mask_functions.py:186, in create_flux_filtering_mask(method, flux_cutoff, tiltx, tilty, modulation_angles, modulation_amp, n_iter, create_summed_image, verbose, verbose_plot, OnSky, **kwargs)
    183 else:
    184     raise ValueError("Invalid method. Use 'tip_tilt_modulation' or 'dm_random'.")
--> 186 fits.writeto(os.path.join(folder_calib, f'summed_pyr_images_3s_pyr{suffix}.fits'), summed_image.astype(np.float32), overwrite=True)
    187 fits.writeto(os.path.join(folder_calib, f'summed_pyr_images_3s_pyr{suffix}_{timestamp}.fits'), summed_image.astype(np.float32), overwrite=True)
    188 if verbose:

File ~/miniconda3/lib/python3.12/site-packages/astropy/io/fits/convenience.py:464, in writeto(filename, data, header, output_verify, overwrite, checksum)
    462 if hdu.is_image and not isinstance(hdu, PrimaryHDU):
    463     hdu = PrimaryHDU(data, header=header)
--> 464 hdu.writeto(
    465     filename, overwrite=overwrite, output_verify=output_verify, checksum=checksum
    466 )

File ~/miniconda3/lib/python3.12/site-packages/astropy/io/fits/hdu/base.py:407, in _BaseHDU.writeto(self, name, output_verify, overwrite, checksum)
    404 from .hdulist import HDUList
    406 hdulist = HDUList([self])
--> 407 hdulist.writeto(name, output_verify, overwrite=overwrite, checksum=checksum)

File ~/miniconda3/lib/python3.12/site-packages/astropy/io/fits/hdu/hdulist.py:1033, in HDUList.writeto(self, fileobj, output_verify, overwrite, checksum)
   1029 mode = FILE_MODES[fileobj_mode(fileobj)] if isfile(fileobj) else "ostream"
   1031 # This can accept an open file object that's open to write only, or in
   1032 # append/update modes but only if the file doesn't exist.
-> 1033 fileobj = _File(fileobj, mode=mode, overwrite=overwrite)
   1034 hdulist = self.fromfile(fileobj)
   1035 try:

File ~/miniconda3/lib/python3.12/site-packages/astropy/io/fits/file.py:218, in _File.__init__(self, fileobj, mode, memmap, overwrite, cache, use_fsspec, fsspec_kwargs, decompress_in_memory)
    216     self._open_fileobj(fileobj, mode, overwrite)
    217 elif isinstance(fileobj, (str, bytes)):
--> 218     self._open_filename(fileobj, mode, overwrite)
    219 else:
    220     self._open_filelike(fileobj, mode, overwrite)

File ~/miniconda3/lib/python3.12/site-packages/astropy/io/fits/file.py:651, in _File._open_filename(self, filename, mode, overwrite)
    648 ext = os.path.splitext(self.name)[1]
    650 if not self._try_read_compressed(self.name, magic, mode, ext=ext):
--> 651     self._file = open(self.name, IO_FITS_MODES[mode])
    652     self.close_on_error = True
    654 # Make certain we're back at the beginning of the file
    655 # BZ2File does not support seek when the file is open for writing, but
    656 # when opening a file for write, bz2.BZ2File always truncates anyway.

FileNotFoundError: [Errno 2] No such file or directory: '/home/ristretto-dao/optlab-master/PROJECTS_3/RISTRETTO/Banc AO/outputs/Calibration_files_papyrus_2026/summed_pyr_images_3s_pyr.fits'

In [13]: 