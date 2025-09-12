import numpy as np
import time
import cupy as cp

slopes = np.random.rand(999)
RM = np.random.rand(999, 195)

RM_cp = cp.asanyarray(RM)
for i in range(100):
    t1 = time.time()
    slopes_cp = cp.asanyarray(slopes)
    #modes = slopes @ RM
    modes = cp.matmul(slopes_cp, RM_cp)
    t2 = time.time() - t1
    print(t2*1000, 'ms')