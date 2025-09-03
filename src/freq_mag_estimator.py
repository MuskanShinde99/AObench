import dao
import numpy as np
import time
from dd_utils import *
import toml
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

with open('control_config.toml', 'r') as f:
    config = toml.load(f)
with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)

fs = dao.shm(shm_path['control']['fs']).get_data()[0][0]
sem_nb = config['sem_nb']['fft']
n_fft = config['visualizer']['n_fft']
n_modes = config['common']['n_modes']

modes_buf_shm = dao.shm(shm_path['control']['modes_buf'])
commands_buf_shm = dao.shm(shm_path['control']['commands_buf'])
pol_buf_shm = dao.shm(shm_path['control']['pol_buf'])

modes_fft_shm = dao.shm(shm_path['control']['modes_fft'])
commands_fft_shm = dao.shm(shm_path['control']['commands_fft'])
pol_fft_shm = dao.shm(shm_path['control']['pol_fft'])
f_shm = dao.shm(shm_path['control']['f'])

update_rate = config['freq_mag_estimator']['update_rate']
closed_loop_flag_shm = dao.shm(shm_path['control']['closed_loop_flag'])

time_start = time.perf_counter()

while True:
    # if (time.perf_counter() - time_start > update_rate and closed_loop_flag_shm.get_data(check=False, semNb=sem_nb)):
    if (time.perf_counter() - time_start > update_rate):
        pol_buf = pol_buf_shm.get_data(check = True, semNb=sem_nb)
        res_buf = modes_buf_shm.get_data(check = True, semNb=sem_nb)
        command_buf = commands_buf_shm.get_data(check = True, semNb=sem_nb)

        if (pol_buf.any() and res_buf.any() and command_buf.any()):

            pol_fft, f, _ = compute_fft_mag_welch(pol_buf, n_fft, fs)
            res_fft, f, _ = compute_fft_mag_welch(res_buf, n_fft, fs)
            command_fft, f, _ = compute_fft_mag_welch(command_buf, n_fft, fs)

            if n_modes == 1:
                pol_fft = pol_fft[:,np.newaxis]
                res_fft = res_fft[:,np.newaxis]
                command_fft = command_fft[:,np.newaxis]
            f = f[:,np.newaxis]
            pol_fft_shm.set_data((pol_fft[1:,:]).astype(np.float32))
            modes_fft_shm.set_data((res_fft[1:,:]).astype(np.float32))
            commands_fft_shm.set_data((command_fft[1:,:]).astype(np.float32))
            f_shm.set_data(f[1:,:].astype(np.float32))

        time_start = time.perf_counter()
