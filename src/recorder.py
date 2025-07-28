import dao
import numpy as np
import toml
from toml_file_updater import TomlFileUpdater
import os
from datetime import datetime
import time 
import astropy.io.fits as fits
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0
# list to record
# slopes
# modes
# command
# M2V
# M2S
# selected mode
# dd order 
# dd n fft
# powers of 2
# n_modes low
# n_modes high
# n_modes controlled 

 
print('starting record')

folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
parent_dir = "record/"
full_path = os.path.join(parent_dir, folder_name)

os.makedirs(full_path, exist_ok=True)

with open('control_config.toml', 'r') as f:
    config = toml.load(f)
with open('shm_path.toml', 'r') as f:
    shm_path = toml.load(f)

sem_nb = config['sem_nb']['rec']
n_modes = config['common']['n_modes']
n_voltages = config['common']['n_voltages']
fs = dao.shm(shm_path['control']['fs']).get_data()[0][0]
record_time = dao.shm(shm_path['control']['record_time']).get_data()[0][0]
modes_shm = dao.shm(shm_path['control']['modes'])
dm_shm = dao.shm(shm_path['control']['dm'])
slopes_shm = dao.shm(shm_path['control']['slopes'])
telemetry_shm = dao.shm(shm_path['control']['telemetry'])

n_fft = dao.shm(shm_path['control']['n_fft']).get_data()[0][0]
controller_select = dao.shm(shm_path['control']['controller_select']).get_data()[0][0]
gain_margin = dao.shm(shm_path['control']['gain_margin']).get_data()[0][0]
dd_order_low = dao.shm(shm_path['control']['dd_order_low']).get_data()[0][0]
dd_order_high = dao.shm(shm_path['control']['dd_order_high']).get_data()[0][0]
dd_update_rate_low = dao.shm(shm_path['control']['dd_update_rate_low']).get_data()[0][0]
dd_update_rate_high = dao.shm(shm_path['control']['dd_update_rate_high']).get_data()[0][0]
n_modes_int = dao.shm(shm_path['control']['n_modes_int']).get_data()[0][0]
n_modes_dd_low = dao.shm(shm_path['control']['n_modes_dd_low']).get_data()[0][0]
n_modes_dd_high = dao.shm(shm_path['control']['n_modes_dd_high']).get_data()[0][0]
record_time = dao.shm(shm_path['control']['record_time']).get_data()[0][0]
delay = dao.shm(shm_path['control']['delay']).get_data()[0][0]
M2V = dao.shm(shm_path['control']['M2V']).get_data()
S2M = dao.shm(shm_path['control']['S2M']).get_data()

norm_flux_pyr_img_shm = dao.shm(shm_path['norm_flux_pyr_img'])

record_its = int(record_time*fs)

results_file = TomlFileUpdater(os.path.join(full_path, "results.toml"))
results_file.add('n_fft',n_fft)

match controller_select:
    case 0:
        results_file.add('control mode','int')
    case 1:
        results_file.add('control mode','dd')
        results_file.add('n_fft',n_fft)
        results_file.add('gain_margin',gain_margin)
        results_file.add('dd_order_low',dd_order_low)
        results_file.add('dd_order_high',dd_order_high)
        results_file.add('dd_update_rate_low',dd_update_rate_low)
        results_file.add('dd_update_rate_high',dd_update_rate_high)
        results_file.add('n_modes_dd_low',n_modes_dd_low)
        results_file.add('n_modes_dd_high',n_modes_dd_high)
    case 2:
        results_file.add('control mode','omgi')
        results_file.add('n_fft',n_fft)
        results_file.add('gain_margin',gain_margin)

results_file.add('delay',delay)
results_file.add('n modes controlled',n_modes_int)
results_file.add('record time',record_time)

modes_buf = np.zeros((record_its,n_modes))
command_buf = np.zeros((record_its,n_modes))
voltages_buf = np.zeros((record_its,n_voltages))
pyr_flux_buf = np.zeros((record_its,1))

for i in range(record_its):

    # modes = modes_shm.get_data(check=True, semNb=sem_nb).squeeze()
    telemetry = telemetry_shm.get_data(check=True, semNb=sem_nb)
    modes = telemetry[0, :]
    command =  telemetry[1, :]
    voltages = dm_shm.get_data(check=False, semNb=sem_nb).squeeze()
    pyr_flux = norm_flux_pyr_img_shm.get_data(check=False, semNb=sem_nb).squeeze()

    modes_buf = np.roll(modes_buf, -1, axis=0)
    voltages_buf = np.roll(voltages_buf, -1, axis=0)
    command_buf = np.roll(command_buf, -1, axis=0)
    pyr_flux_buf = np.roll(pyr_flux_buf, -1, axis=0)

    modes_buf[-1, :] = modes
    voltages_buf[-1, :] = voltages
    command_buf[-1, :] = command
    pyr_flux_buf[-1, :] = pyr_flux


rms = np.mean(np.sum(np.square(modes_buf),axis=1))
results_file.add('rms',rms)

fits.writeto(os.path.join(full_path, "modes.fits"), modes_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "voltages.fits"), voltages_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "commands.fits"), command_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "pyr_fluxes.fits"), pyr_flux_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "M2V.fits"), M2V, overwrite = True)
fits.writeto(os.path.join(full_path, "S2M.fits"), S2M, overwrite = True)

results_file.save()

print('print record done')