# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:13:17 2024

@author: RISTRETTO
"""
import numpy as np
import pylab as plt
from pypylon import pylon
import tqdm

#%%
def pylonGrab(camera, Nimg, progressBar=True):

    dit = camera.ExposureTime()
    runtime = dit*Nimg*1e-6
    if (progressBar) and (runtime > 5) and (Nimg>1):
        progressBar = True
    else:
        progressBar = False        

    if progressBar:
        pbar = tqdm.tqdm(total = Nimg, desc='Grabbing...', leave=False)

    img = 0
    camera.StartGrabbingMax(Nimg)
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
        if grabResult.GrabSucceeded():
            if progressBar:
                pbar.update(1)
            img += np.array(grabResult.Array.astype('float64'))
        grabResult.Release()
    img /= Nimg

    return img


#%%
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.PixelFormat.SetValue('Mono10')


# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
    camera.Width.SetValue(new_width)

Nimg = 10
img = pylonGrab(camera, Nimg)

camera.Close()

#%%
# ** In this example, we define two regions in horizontal direction
# that will be transmitted as a single image. **

# Enable the ability to configure multiple columns
camera.BslMultipleROIColumnsEnable.Value = True
# Select column 1
camera.BslMultipleROIColumnSelector.Value = "Column1"
# The first region should have a horizontal offset of 100 and a width of 300 pixels
camera.BslMultipleROIColumnOffset.Value = 100
camera.BslMultipleROIColumnSize.Value = 300
# Select column 2
camera.BslMultipleROIColumnSelector.Value = "Column2"
# The second region should have a horizontal offset of 500 and a width of 400 pixels
camera.BslMultipleROIColumnOffset.Value = 500
camera.BslMultipleROIColumnSize.Value = 400

# We only need one row, so disable the ability to configure multiple rows
camera.BslMultipleROIRowsEnable.Value = False
# Both regions should have a vertical offset of 200 and a height of 500
camera.OffsetY.Value = 200
camera.Height.Value = 500