# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:56:47 2024

@author: RISTRETTO
"""
import os
import sys
from pathlib import Path

from src.config import config
from DEVICES_3.Thorlabs.MCLS1 import mcls1

ROOT_DIR = config.root_dir

channel = 1
state = 0 # 0 to turn OFF, 1 to turn ON
las = mcls1("/dev/ttyUSB0")
las.set_channel(channel)
las.enable(state) 

#las.set_current(55) #55mA is a good value for pyramid images
