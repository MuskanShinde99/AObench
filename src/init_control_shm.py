import toml
import dao
import numpy as np
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0
with open('control_config.toml', 'r') as f:
    config = toml.load(f)
with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)

# load parameters
n_fft_display = config['visualizer']['n_fft']
buf_size = config['visualizer']['buf_size']
max_order = config['optimizer']['max_order']
n_modes = config['common']['n_modes']
gain = config['integrator']['gain']
dm_shm = dao.shm(shm_path['control']['dm'])

# create shm
# time domain data
modes_buf = np.zeros((buf_size,n_modes),np.float32)
telemetry = np.zeros((2,n_modes),np.float32)
telemetry_ts = np.zeros((2,1),np.float64)
commands_buf = np.zeros((buf_size,n_modes),np.float32)
pol_buf = np.zeros((buf_size,n_modes),np.float32)
state_mat = np.zeros((2*max_order+1, n_modes),np.float32)
K_mat = np.zeros((2*max_order+1, n_modes),np.float32)
K_mat[0,:] = gain
K_mat[max_order+1, :] = 0.99

# frequency domain data
modes_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
commands_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
pol_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
f = np.ones((int(n_fft_display/2),1),np.float32)
t = np.zeros((buf_size,1),dtype = np.float32)

# loop variables
delay = np.array([[config['common']['delay']]],dtype = np.float32)
fs = np.array([[config['common']['fs']]],dtype = np.float32)
latency = np.zeros((1,1),dtype = np.float32)

closed_loop_flag = np.zeros((1,1),dtype = np.uint32)

n_modes_dd_high = np.zeros((1,1),dtype = np.uint32)
n_modes_dd_low = np.zeros((1,1),dtype = np.uint32)
n_modes_int = np.zeros((1,1),dtype = np.uint32)
record_time = np.zeros((1,1),dtype = np.float32)
mutex_opti = np.zeros((1,1),dtype = np.uint32)
reset_flag = np.zeros((1,1),dtype = np.uint32)
modes = np.zeros((n_modes,1),dtype=np.float32)

dd_update_rate_high = np.array([[np.inf]],dtype = np.float32)
dd_update_rate_low = np.array([[np.inf]],dtype = np.float32)

gain_margin = np.array([[1.2]],dtype = np.float32)
wait_time = np.array([[config['calibration']['wait_time']]],dtype = np.float32)
n_fft_optimizer = np.array([[config['optimizer']['n_fft']]],dtype = np.uint32)
dd_order_high = np.array([[5]],dtype = np.uint32)
dd_order_low = np.array([[20]],dtype = np.uint32)

slopes_shm = dao.shm(shm_path['control']['slopes'])
slopes = slopes_shm.get_data()
n_slopes = slopes.shape[0]
S2M =np.zeros((n_modes,n_slopes),dtype=np.float32)

controller_select = np.zeros((1,1),dtype = np.uint32)
pyramid_select = np.zeros((1,1),dtype = np.uint32)

n_fft_max = config['optimizer']['n_fft_max']

S_dd = np.ones((n_fft_max,n_modes),dtype=np.float32)
S_omgi = np.ones((n_fft_max,n_modes),dtype=np.float32)
S_int = np.ones((n_fft_max,1),dtype=np.float32)
f_opti = np.ones((n_fft_max,1),dtype=np.float32)
f_opti[:n_fft_optimizer[0][0]]= np.linspace(0.1,fs[0][0]/2,n_fft_optimizer[0][0])[:,np.newaxis]

n_act = dm_shm.get_data().shape[0]
flat = np.zeros((n_act,1),dtype = np.float32)



dao.shm(shm_path['control']['modes_buf'],modes_buf)
dao.shm(shm_path['control']['commands_buf'],commands_buf)
dao.shm(shm_path['control']['pol_buf'],pol_buf)
dao.shm(shm_path['control']['t'],t)

dao.shm(shm_path['control']['state_mat'],state_mat)
dao.shm(shm_path['control']['K_mat_int'],K_mat)
dao.shm(shm_path['control']['K_mat_dd'],K_mat)
dao.shm(shm_path['control']['K_mat_omgi'],K_mat)


dao.shm(shm_path['control']['modes_fft'],modes_fft)
dao.shm(shm_path['control']['commands_fft'],commands_fft)
dao.shm(shm_path['control']['pol_fft'],pol_fft)
dao.shm(shm_path['control']['f'],f)


dao.shm(shm_path['control']['delay'],delay)
dao.shm(shm_path['control']['fs'],fs)
dao.shm(shm_path['control']['latency'],latency)

dao.shm(shm_path['control']['closed_loop_flag'],closed_loop_flag)

dao.shm(shm_path['control']['n_modes_dd_high'],n_modes_dd_high)
dao.shm(shm_path['control']['n_modes_dd_low'],n_modes_dd_low)
dao.shm(shm_path['control']['n_modes_int'],n_modes_int)
dao.shm(shm_path['control']['mutex_opti'],mutex_opti)
dao.shm(shm_path['control']['reset_flag'],reset_flag)

dao.shm(shm_path['control']['dd_update_rate_high'],dd_update_rate_high)
dao.shm(shm_path['control']['dd_update_rate_low'],dd_update_rate_low)

dao.shm(shm_path['control']['dd_order_high'],dd_order_high)
dao.shm(shm_path['control']['dd_order_low'],dd_order_low)

dao.shm(shm_path['control']['S2M'],S2M)
if config['common']['use_own_modes'] == 1:
    dao.shm(shm_path['control']['modes_4'],modes)

dao.shm(shm_path['control']['controller_select'],controller_select)
dao.shm(shm_path['control']['pyramid_select'],pyramid_select)
dao.shm(shm_path['control']['gain_margin'],gain_margin)

dao.shm(shm_path['control']['wait_time'],wait_time)
dao.shm(shm_path['control']['n_fft'],n_fft_optimizer)
dao.shm(shm_path['control']['record_time'],record_time)

dao.shm(shm_path['control']['S_dd'],S_dd)
dao.shm(shm_path['control']['S_omgi'],S_omgi)
dao.shm(shm_path['control']['S_int'],S_int)
dao.shm(shm_path['control']['f_opti'],f_opti)

dao.shm(shm_path['control']['telemetry'],telemetry)
dao.shm(shm_path['control']['telemetry_ts'],telemetry_ts)