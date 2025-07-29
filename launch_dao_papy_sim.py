import subprocess
import signal
import os
import time
from scipy.io import loadmat
import dao
import numpy as np

# List of script commands
scripts = [
    "daoPapyrusCreateShm.py",
    "daoPapyrusCreateSHM.py",
    "daoPapyrusSimStart -b 1",
    "papyCtrl.py",
    "daoDmDisp.py -s dmCmd -m dm241Map",
    "daoBarRTD.py -s /tmp/papyrus_modes.im.shm",
]

# List to store process objects
processes = []

def cleanup():
    print("\nCleaning up...")
    for proc in processes:
        try:
            print(f"Terminating {proc.pid}...")
            os.kill(proc.pid, signal.SIGTERM)  # Send SIGTERM to the process
            print(f"Process {proc.pid} terminated.")
        except Exception as e:
            print(f"Error terminating process {proc.pid}: {e}")

    # Kill all tmux sessions
    try:
        print("Killing all tmux sessions...")
        subprocess.run(["tmux", "kill-server"], check=True)
        print("All tmux sessions terminated.")
    except FileNotFoundError:
        print("tmux not found. Ensure it is installed and in the PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error while killing tmux sessions: {e}")

    print("Cleanup complete.")

# Run the scripts in the background
for script in scripts:
    try:
        print(f"Running {script}...")
        command = script.split()
        proc = subprocess.Popen(command)  # Start the script in the background
        processes.append(proc)  # Keep track of the process
        print(f"{script} started with PID {proc.pid}.")
    except FileNotFoundError:
        print(f"{script} not found. Ensure it is in the PATH or the current directory.")

try:
    print("Press Ctrl+C to terminate all jobs and tmux sessions...")
    while True:  
        time.sleep(1)

except KeyboardInterrupt:
    print("\nCtrl+C detected!")
    cleanup()
