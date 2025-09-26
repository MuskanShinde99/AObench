import os
from pathlib import Path
from datetime import datetime

import dao
import numpy as np
import matplotlib.pyplot as plt
from src.config import config

# ---------- Output setup ----------
folder_cred3 = config.folder_cred3
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = folder_cred3 / f"cred3_missed_frames_{timestamp}.png"

# ---------- Data acquisition ----------
cred3_frame_counter_shm = dao.shm('/tmp/cred3_frame_counter.im.shm')

difference_array = []
start_count = cred3_frame_counter_shm.get_data()[0, 0]
print('start count', start_count)

for i in np.arange(1, 50000, 1):
    current_count = cred3_frame_counter_shm.get_data(check=True)[0, 0]
    difference_array.append(current_count)

# ---------- Plot & save ----------
plt.figure()
diff_vals = np.diff(difference_array) if len(difference_array) > 1 else np.array([])
plt.plot(diff_vals)
plt.title(f"CRED3 missed frames counter")
plt.xlabel("Index")
plt.ylabel(" Number of frames")
plt.grid(True)

# Save first, then (optionally) show
plt.savefig(str(outfile), dpi=150, bbox_inches='tight')
print(f"Saved plot to: {outfile}")

plt.show()
