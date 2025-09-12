import dao
import numpy as np
import matplotlib.pyplot as plt

cred3_frame_counter_shm = dao.shm('/tmp/cred3_frame_counter.im.shm')

difference_array = []
start_count = cred3_frame_counter_shm.get_data()[0,0]
print('start count', start_count)

for i in np.arange(1, 50000, 1):

    current_count = cred3_frame_counter_shm.get_data(check=True)[0,0]
    #difference = current_count - start_count
    #print('difference', difference)

    difference_array.append(current_count)
    #start_count = current_count


plt.figure()
plt.plot(np.diff(difference_array))
plt.show()
    