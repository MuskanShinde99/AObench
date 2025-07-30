import toml
import dao
import numpy as np
from dd_utils import *
import dd4ao
import time
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

with open('control_config.toml', 'r') as f:
    config = toml.load(f)

with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)

sem_nb = config['sem_nb']['calib']
modes_shm = dao.shm(shm_path['control']['modes'])
n_modes = config['common']['n_modes']

t_shm = dao.shm(shm_path['control']['t'])
dm_shm = dao.shm(shm_path['control']['dm'])
latency_shm = dao.shm(shm_path['control']['latency'])
delay_shm = dao.shm(shm_path['control']['delay'])
fs_shm = dao.shm(shm_path['control']['fs'])

buf_size = config['visualizer']['buf_size']

M2V = dao.shm(shm_path['control']['M2V']).get_data()
M2V = M2V[:,:n_modes]

amp = config['calibration']['amp']

n_iter = 1000

latency_array = np.zeros(n_iter)
frequency_array = np.zeros(n_iter)

eps = amp/2

command = np.zeros((n_modes,1),np.float32)

for i in range(n_iter):

    command[0] = amp
    voltage = M2V@command
    # time.sleep(0.001)
    dm_shm.set_data(voltage.astype(np.float32))

    while (modes_shm.get_data(check = True, semNb = sem_nb))[0] < command[0] - eps:
        pass

    command[0] = -amp
    voltage = M2V@command
    time_start = time.perf_counter()
    # time.sleep(0.002)
    dm_shm.set_data(voltage.astype(np.float32))
    

    while (modes_shm.get_data(check = True, semNb = sem_nb))[0] > command[0] + eps:
        pass

    latency_array[i] = time.perf_counter() - time_start


mean_latency = np.mean(latency_array)
max_latency_jitter = np.max(latency_array)-np.min(latency_array)
std_latency = np.std(latency_array)

for i in range(n_iter):
    time_start = time.perf_counter()
    modes_shm.get_data(check = True, semNb = sem_nb)
    frequency_array[i] = 1/(time.perf_counter() - time_start)

mean_frequency = np.mean(frequency_array)
max_frequency_jitter = np.max(frequency_array)-np.min(frequency_array)
std_frequency = np.std(frequency_array)
mean_delay = mean_frequency*mean_latency

print(f"mean latency = {mean_latency:.4f} [s], max latency jitter = {max_latency_jitter:.4f} [s], std latency = {std_latency:.4f} [s] ")
print(f"mean frequency = {mean_frequency:.4f} [Hz], max frequency jitter = {max_frequency_jitter:.4f} [Hz], std frequency = {std_frequency:.4f} [Hz] ")
print(f"mean delay = {mean_delay:.4f} [frames]")

command[0] = 0
voltage = M2V@command
dm_shm.set_data(voltage.astype(np.float32))



t = np.arange(0,buf_size,dtype = np.float32)/mean_frequency
t = t[:,np.newaxis]
t_shm.set_data(t)

latency_shm.set_data(np.array([[mean_latency]],np.float32))
delay_shm.set_data(np.array([[mean_delay]],np.float32))
fs_shm.set_data(np.array([[mean_frequency]],np.float32))
