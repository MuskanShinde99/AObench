import toml
import numpy as np
from dd_utils import *
import dd4ao
import time
import sys
import contextlib
import os
import dao 
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0


with open('config.toml', 'r') as f:
    config = toml.load(f)
with open('fixed_params.toml', 'r') as f:
    fixed_params = toml.load(f)


sem_nb = config['sem_nb']['optimizer']
max_order = config['optimizer']['max_order']
order = 1
update_rate = config['optimizer']['update_rate']
n_fft = int(dao.shm(fixed_params['shm_path']['n_fft']).get_data(check=False, semNb=sem_nb)[0][0])
bandwidth = config['optimizer']['bandwidth']
fs = dao.shm(fixed_params['shm_path']['fs']).get_data(check=False, semNb=sem_nb)[0][0]
delay = dao.shm(fixed_params['shm_path']['delay']).get_data(check=False, semNb=sem_nb)[0][0]
gain_margin = dao.shm(fixed_params['shm_path']['gain_margin']).get_data(check=False, semNb=sem_nb)[0][0]
f_shm = dao.shm(fixed_params['shm_path']['f'])
pol_fft_shm = dao.shm(fixed_params['shm_path']['pol_fft'])
K_mat_shm = dao.shm(fixed_params['shm_path']['K_mat_omgi'])
pol_buf_shm = dao.shm(fixed_params['shm_path']['pol_buf'])
n_modes = dao.shm(fixed_params['shm_path']['n_modes_int']).get_data(check=False, semNb=sem_nb)[0][0]

S_shm = dao.shm(fixed_params['shm_path']['S_omgi']) 
f_opti_shm = dao.shm(fixed_params['shm_path']['f_opti'])
S = S_shm.get_data(check=False, semNb=sem_nb)
K_array  = np.empty(n_modes,dtype = dd4ao.DD4AO)

t_start = time.perf_counter()
training_set = pol_buf_shm.get_data(check=False, semNb=sem_nb)
K_mat = K_mat_shm.get_data(check=False, semNb=sem_nb)

f_p =  f_shm.get_data(check=False, semNb=sem_nb).squeeze()
f = np.linspace(f_p[0],f_p[-1],n_fft)
f_opti_shm.set_data(f.astype(np.float32))
w = 2*np.pi*f
G_resp = G_freq_resp(delay, w, fs)*gain_margin

for i in range(n_modes):
    print(i)
    pol_fft_p = pol_fft_shm.get_data(check=False, semNb=sem_nb)
    pol_fft = np.interp(f,f_p,pol_fft_p[:,i])
    K_array[i] = dd4ao.DD4AO(w, G_resp, pol_fft, order, bandwidth,
                fs, K_mat[0:order+1,i],np.concatenate((np.array([1]),-K_mat[max_order+1:max_order+1+order,i])),Fx=np.array([1,0]), Fy=np.array([1,-0.99]),overshoot_weight=0.0001)
    K_array[i].compute_controller()
    K_mat[:order + 1,i] = K_array[i].num.squeeze()
    K_mat[max_order + 1:max_order+order+1,i] = -K_array[i].den.squeeze()[1:]
    K_mat_shm.set_data(K_mat)
    S[:n_fft,i] = 1/np.abs(K_array[i].S_resp.squeeze())
S_shm.set_data(S)
elapsed_time = time.perf_counter() - t_start
print('Controller optimized in = {:.2f} s'.format(elapsed_time))


# time.sleep(30)


# G = G_tf(2, fs)
# mode = 194

# K = K_array[mode].K
# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_comp_sensitivity(axs, G, K, K, f)


# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_sensitivity(axs, G, K, K, pol_psd[:,mode],f,10)
# plt.show()


