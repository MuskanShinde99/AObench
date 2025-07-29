import subprocess
import signal
import os
import time
from scipy.io import loadmat
import dao
import numpy as np
import astropy.io.fits as pfits

time.sleep(1) # wait for SHM to be created
amp=0.1
turb = loadmat("turbulence_r0_15cm_windSpeed_5_ms_frequency_1000_Hz_seed_1.mat")["phase_screen"]
dmTurb=dao.shm('/tmp/dmCmd03.im.shm')
M2V = dao.shm("/tmp/m2c.im.shm").get_data()
slopes_shm = dao.shm('/tmp/papyrus_slopes.im.shm')
# Infinite loop to wait for Ctrl+C
fs = 100
old_time = time.time()
while True:  
    for k in range(turb.shape[1]):
        new_time = time.time()
        # print(1/(new_time-old_time))
        old_time = new_time

        dmTurb.set_data(3.6e5*turb[:,k].astype(np.float32)*amp)
        slopes_shm.get_data(check=True, semNb=4)
    for k in np.linspace(turb.shape[1]-1,0,turb.shape[1]):
        new_time = time.time()
        # print(1/(new_time-old_time))
        old_time = new_time
        dmTurb.set_data(3.6e5*turb[:,int(k)].astype(np.float32)*amp)
        slopes_shm.get_data(check=True, semNb=4)

