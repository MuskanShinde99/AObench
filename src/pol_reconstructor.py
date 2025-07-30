import dao
import numpy as np
import toml
import time 
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

with open('control_config.toml', 'r') as f:
    config = toml.load(f)
with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)

sem_nb = config['sem_nb']['pol']
max_order = config['optimizer']['max_order']
delay = dao.shm(shm_path['control']['delay']).get_data()[0][0]
# delay = 2
fs = dao.shm(shm_path['control']['fs']).get_data()[0][0]
update_rate = config['visualizer']['update_rate']


telemetry_shm = dao.shm(shm_path['control']['telemetry'])

modes_buf_shm = dao.shm(shm_path['control']['modes_buf'])
commands_buf_shm = dao.shm(shm_path['control']['commands_buf'])
pol_buf_shm = dao.shm(shm_path['control']['pol_buf'])


def pol_reconstruct(command_buff, measurement_buff, delay):
    delay_floor = int(np.floor(delay))
    delay_ceil = int(np.ceil(delay))
    delay_frac,_ = np.modf(delay)
    if delay_ceil == delay_floor:
        pol = measurement_buff[-1,:] + command_buff[-delay_ceil-1,:]
    else:
        pol = measurement_buff[-1, :] + (1 - delay_frac) * command_buff[-delay_floor-1, :] + delay_frac * command_buff[-delay_ceil-1,:]
    return pol

pol_buf = pol_buf_shm.get_data(semNb=sem_nb)
modes_buf = modes_buf_shm.get_data(semNb=sem_nb)
commands_buf = commands_buf_shm.get_data(semNb=sem_nb)

time_start = time.perf_counter()

while True:
    new_time = time.time()
    # print(1/(new_time-old_time))
    old_time = new_time
    telemetry = telemetry_shm.get_data(check=True, semNb=sem_nb)
    modes = telemetry[0, :]
    command =  telemetry[1, :]

    pol_buf = np.roll(pol_buf, -1, axis=0)
    modes_buf = np.roll(modes_buf, -1, axis=0)
    commands_buf = np.roll(commands_buf, -1, axis=0)

    modes_buf[-1, :] = modes
    commands_buf[-1, :] = command
    pol = pol_reconstruct(commands_buf, modes_buf, delay)
    pol_buf[-1, :] = pol

    if (time.perf_counter() - time_start > update_rate):
        modes_buf_shm.set_data(modes_buf.astype(np.float32))
        commands_buf_shm.set_data(commands_buf.astype(np.float32))
        pol_buf_shm.set_data(pol_buf.astype(np.float32))
        time_start = time.perf_counter()