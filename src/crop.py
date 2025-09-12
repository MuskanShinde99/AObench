import numpy as np
import dao
cred_shm = dao.shm('/tmp/cred3.im.shm')
cred_size_shape = cred_shm.get_data(check = False).shape
cred_cropped = np.zeros((cred_size_shape[0]-1,cred_size_shape[1]),np.uint16)
cred_cropped_shm = dao.shm('/tmp/cred3_cropped.im.shm', cred_cropped)
cred_frame_counter = np.zeros((1, 1), dtype=np.uint16)
cred_frame_counter_shm = dao.shm('/tmp/cred3_frame_counter.im.shm', cred_frame_counter)

start_count = cred_shm.get_data()[0,0]
print('start count', start_count)

while True:
    cred = cred_shm.get_data(check = True, semNb = 6)
    cred_cropped = cred[1:,:]
    cred_cropped_shm.set_data(cred_cropped)
    current_count = cred[0,0]
    cred_frame_counter_shm.set_data(np.array([[current_count-start_count]]))