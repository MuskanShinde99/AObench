

import dao
import numpy as np
import toml
import time 
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0


with open('shm_path_control.toml', 'r') as f:
    shm_path = toml.load(f)


dm_shm = dao.shm(shm_path['control']['dm'])

dm_shm.set_data(dm_shm.get_data(check = False)*0)

dm_shm = dao.shm('/tmp/commands.im.shm')

dm_shm.set_data(dm_shm.get_data(check = False)*0)