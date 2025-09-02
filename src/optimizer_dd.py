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

update_rate = config['optimizer']['update_rate']
n_fft = int(dao.shm(fixed_params['shm_path']['n_fft']).get_data(check=False, semNb=sem_nb)[0][0])
bandwidth = config['optimizer']['bandwidth']
fs = dao.shm(fixed_params['shm_path']['fs']).get_data(check=False, semNb=sem_nb)[0][0]
delay = dao.shm(fixed_params['shm_path']['delay']).get_data(check=False, semNb=sem_nb)[0][0]
# delay = 3
K_mat_shm = dao.shm(fixed_params['shm_path']['K_mat_dd'])
pol_buf_shm = dao.shm(fixed_params['shm_path']['pol_buf'])
n_modes = dao.shm(fixed_params['shm_path']['n_modes_dd']).get_data(check=False, semNb=sem_nb)[0][0]
order = dao.shm(fixed_params['shm_path']['dd_order']).get_data(check=False, semNb=sem_nb)[0][0]
gain_margin = dao.shm(fixed_params['shm_path']['gain_margin']).get_data(check=False, semNb=sem_nb)[0][0]
f_shm = dao.shm(fixed_params['shm_path']['f'])
pol_fft_shm = dao.shm(fixed_params['shm_path']['pol_fft'])

S_shm = dao.shm(fixed_params['shm_path']['S_dd']) 
f_opti_shm = dao.shm(fixed_params['shm_path']['f_opti'])
mutex_opti_shm = dao.shm(fixed_params['shm_path']['mutex_opti'])

print("waiting for mutex")
# while(mutex_opti_shm.get_data(check=False, semNb=sem_nb)[0][0]):
#     time.sleep(0.01)
mutex_opti_shm.set_data(np.ones((1,1),np.uint32))
print("done waiting for mutex")
print("starting optimization")

S = S_shm.get_data(check=False, semNb=sem_nb)
K_array  = np.empty(n_modes,dtype = dd4ao.DD4AO)

t_start = time.perf_counter()
training_set = pol_buf_shm.get_data(check=False, semNb=sem_nb)
K_mat = K_mat_shm.get_data(check=False, semNb=sem_nb)

f_p =  f_shm.get_data(check=False, semNb=sem_nb).squeeze()
f = np.linspace(f_p[0],f_p[-1],n_fft)

f_opti_shm.set_data(f[:,np.newaxis].astype(np.float32)) 
w = 2*np.pi*f
# delay += 0.4
G_resp = G_freq_resp(delay, w, fs)*gain_margin

# G_resp = freqresp(G_tf(delay,fs),w)*gain_margin
start = 7
powers = powers_of_two_between(start,n_modes)
optimization_indexes = np.concatenate((np.arange(start),powers))
n_optmization = optimization_indexes.shape[0]

pol_fft_p = pol_fft_shm.get_data(check=False, semNb=sem_nb) 
pol_fft = np.zeros((n_fft,n_modes))  
for i in range(n_modes):
    pol_fft[:,i] = np.interp(f,f_p,pol_fft_p[:,i])

optimization_indexes_wide = np.searchsorted(optimization_indexes, np.arange(n_modes), side='right') - 1
pol_fft_avg = np.zeros((pol_fft.shape[0], n_optmization))

# Vectorized accumulation (like a histogram with weights)
np.add.at(pol_fft_avg, (slice(None), optimization_indexes_wide), pol_fft)

# Compute bin counts: how many modes per optimization bin
bin_counts = list(np.diff(optimization_indexes))
bin_counts.append(n_modes - optimization_indexes[-1])
bin_counts = np.array(bin_counts)

# Normalize each bin
pol_fft_avg /= bin_counts


for i in range(n_optmization):
    print(i)
    K_array[i] = dd4ao.DD4AO(w, G_resp, pol_fft_avg[:,i], order,fs, n_iter = 10000, tol = 1e-2,high_freq_u_lim=True)
    K_array[i].compute_controller()
    
for i in range(n_modes):
    K_mat[:order + 1,i] = K_array[optimization_indexes_wide[i]].num.squeeze()
    K_mat[max_order + 1:max_order+order+1,i] = -K_array[optimization_indexes_wide[i]].den.squeeze()[1:]
    K_mat_shm.set_data(K_mat)
    S[:n_fft,i] = 1/np.abs(K_array[optimization_indexes_wide[i]].S_resp.squeeze())

S_shm.set_data(S)
elapsed_time = time.perf_counter() - t_start
print('Controller optimized in = {:.2f} s'.format(elapsed_time))
mutex_opti_shm.set_data(np.zeros((1,1),np.uint32))

# time.sleep(30)


# G = G_tf(delay, fs)
# for i in range(n_modes):
#     if(check_K_stability(K_array[i].K,G)>1):
#         print(i)
#         print("unstable")
#         stop

# mode = 6

# K = K_array[mode].K
# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_comp_sensitivity(axs, G, K, K, f)


# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_sensitivity(axs, G, K, K, np.interp(f,f_p,pol_fft_p[:,mode]),f,10)
# # plt.show()

# mode = 1 
# K = K_array[mode].K
# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_comp_sensitivity(axs, G, K, K, f)


# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_sensitivity(axs, G, K, K, np.interp(f,f_p,pol_fft_p[:,mode]),f,10)
# plt.show()

# print(check_K_stability(K_array[mode].K,G))
