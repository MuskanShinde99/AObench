import dao
import numpy as np
import toml
import time 
import datetime
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

with open('control_config.toml', 'r') as f:
    config = toml.load(f)
with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)

n_modes = config['common']['n_modes']
sem_nb = config['sem_nb']['hrtc']
max_order = config['optimizer']['max_order']
max_voltage = config['hrtc']['max_voltage']
modes_3_shm = dao.shm(shm_path['control']['modes_3'])
modes_4_shm = dao.shm(shm_path['control']['modes_4'])
dm_shm = dao.shm(shm_path['control']['dm'])
K_mat_int_shm = dao.shm(shm_path['control']['K_mat_int'])
K_mat_dd_shm = dao.shm(shm_path['control']['K_mat_dd'])
K_mat_omgi_shm = dao.shm(shm_path['control']['K_mat_omgi'])
closed_loop_flag_shm = dao.shm(shm_path['control']['closed_loop_flag'])
controller_select_shm = dao.shm(shm_path['control']['controller_select'])
pyramid_select_shm = dao.shm(shm_path['control']['pyramid_select'])
n_modes_dd_high_shm = dao.shm(shm_path['control']['n_modes_dd_high'])
n_modes_controlled_shm = dao.shm(shm_path['control']['n_modes_int'])
telemetry_shm = dao.shm(shm_path['control']['telemetry']) 
telemetry_ts_shm = dao.shm(shm_path['control']['telemetry_ts']) 
reset_flag_shm = dao.shm(shm_path['control']['reset_flag']) 
fs = dao.shm(shm_path['control']['fs']).get_data(check=False, semNb=sem_nb)[0][0]

M2V = dao.shm(shm_path['control']['M2V']).get_data(check=False, semNb=sem_nb)
V2M = np.linalg.pinv(M2V)
state_mat = np.zeros((2*max_order+1, n_modes),np.float32)
telemetry = np.zeros((2,n_modes),np.float32)
telemetry_ts = np.zeros((2,1),np.float32)
epoch = np.datetime64('1970-01-01T00:00:00', 'us')

old_time = time.time()
print_rate = 1 # [s]
time_at_last_print = time.perf_counter()
time_at_last_error_print = time.perf_counter()
last_loop_time = time.perf_counter()
counter = 0
read_time = 0
computation_time = 0
write_time = 0
wfs_time = 0
loop_time = 0
jitter_array = np.zeros(int(print_rate*fs*2))

K_mat = K_mat_int_shm.get_data(check=False, semNb=sem_nb)

while True:

    loop_time += time.perf_counter() - last_loop_time
    jitter_array[counter] = time.perf_counter() - last_loop_time
    last_loop_time = time.perf_counter()
    

    start_wfs_time = time.perf_counter()
    match pyramid_select_shm.get_data(check=False, semNb=sem_nb)[0][0]:
        case 0:
            modes = modes_4_shm.get_data(check=True, semNb=sem_nb).squeeze()
            modes_ts = np.datetime64(modes_4_shm.get_timestamp(), 'us')
        case 1:
            modes = modes_3_shm.get_data(check=True, semNb=sem_nb).squeeze()
            modes_ts = np.datetime64(modes_3_shm.get_timestamp(), 'us')

    wfs_time += time.perf_counter() - start_wfs_time

    start_read_time = time.perf_counter()

    reset_flag = reset_flag_shm.get_data(check=False)
    if reset_flag == 1:
        reset_flag_shm.set_data(np.zeros((1,1),dtype=np.float32))
        state_mat = np.zeros_like(state_mat)
        print("state mat reset")

    n_modes_controlled = n_modes_controlled_shm.get_data(check=False, semNb=sem_nb)[0][0]
    K_mat = K_mat_int_shm.get_data(check=False, semNb=sem_nb)
    match controller_select_shm.get_data(check=False, semNb=sem_nb)[0][0]:
        case 1:
            K_mat_dd = K_mat_dd_shm.get_data(check=False, semNb=sem_nb)    
            n_modes_dd_high = n_modes_dd_high_shm.get_data(check=False, semNb=sem_nb)[0][0]
            K_mat[:,:n_modes_dd_high] = K_mat_dd[:,:n_modes_dd_high]
        case 2:
            K_mat = K_mat_omgi_shm.get_data(check=False, semNb=sem_nb)

    if not closed_loop_flag_shm.get_data(check=False, semNb=sem_nb):
        K_mat = np.zeros_like(K_mat)


    read_time += time.perf_counter() - start_read_time
    start_computation_time = time.perf_counter()

    state_mat[1:, :] = state_mat[0:-1, :]
    state_mat[0, :] = modes
    command_mat = np.multiply(state_mat, K_mat)
    command = np.sum(command_mat, axis=0)
    command[n_modes_controlled:] = 0
    
    voltage = -M2V[:,:] @ command
    if np.isnan(np.sum(voltage)):
        command = np.zeros_like(command)
        voltage = np.zeros_like(voltage)
        if time.perf_counter() - time_at_last_print > print_rate:
            print("unstability detected")
            time_at_last_error_print = time.perf_counter()


    elif np.max(np.abs(voltage)) >= max_voltage:
        voltage = np.clip(voltage,-max_voltage, max_voltage)
        command = V2M@voltage
        if time.perf_counter() - time_at_last_print > print_rate:
            print("saturation detected")
            time_at_last_error_print = time.perf_counter()


    state_mat[max_order, :] = command
    # time.sleep(1e-3)
    computation_time += time.perf_counter() - start_computation_time
    start_write_time = time.perf_counter()


    telemetry[0,:] = modes
    telemetry[1,:] = command
    telemetry_shm.set_data(telemetry.astype(np.float32))
    command_ts = np.datetime64(datetime.datetime.now(), 'us')
    telemetry_ts[0,:] = (modes_ts - epoch) / np.timedelta64(1, 's')
    telemetry_ts[1,:] = (command_ts - epoch) / np.timedelta64(1, 's')
    # telemetry_ts_shm.set_data(telemetry_ts)

    # time.sleep(0.001)
    # time.sleep(0.002*np.random.rand())
    dm_shm.set_data(voltage.astype(np.float32))

    write_time += time.perf_counter() - start_write_time

    counter += 1

    if(time.perf_counter() - time_at_last_print > 1):
        print('Loop rate = {:.2f} Hz'.format(1/(loop_time/counter)))
        print('Loop time  = {:.2f} ms'.format(loop_time/counter*1e3))
        print('WFS time  = {:.2f} ms'.format(wfs_time/counter*1e3))
        print('Read time = {:.2f} ms'.format(read_time/counter*1e3))
        print('Computation_time = {:.2f} ms'.format(computation_time/counter*1e3))
        print('Write time  = {:.2f} ms'.format(write_time/counter*1e3))
        print('Jitter std  = {:.2f} ms'.format(np.std(jitter_array[:counter])*1e3))
        print('Max loop time  = {:.2f} ms'.format(np.max(jitter_array[:counter]*1e3)))
        print('Framed missed  = {:.2f} '.format(np.sum(jitter_array[:counter]>2/fs)))
        counter = 0
        read_time = 0
        computation_time = 0
        write_time = 0
        loop_time = 0
        wfs_time = 0
        time_at_last_print = time.perf_counter()
