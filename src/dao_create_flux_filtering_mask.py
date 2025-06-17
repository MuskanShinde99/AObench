#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:19:48 2025

@author: laboptic
"""

import numpy as np
from pypylon import pylon
from DEVICES_3.Basler_Pylon.test_pylon import *
from hcipy import *
import time
from matplotlib import pyplot as plt
import src.dao_setup as dao_setup # Import the setup file
from src.tilt import *
from src.utils import *
import matplotlib.colors as mcolors



def create_flux_filtering_mask(modulation_angles, modulation_amp, verbose=False, verbose_plot=False, **kwargs):
    """
    Create a mask based on the modulation angles and flux cutoff.

    Parameters:
    - modulation_angles: list of angles for modulation
    - modulation_amp: modulation amplitude
    - verbose: if True, print info messages
    - verbose_plot: if True, update the displayed image dynamically
    - kwargs: optional overrides for:
        - camera: camera object 
        - slm: spatial light modulator object 
        - data_pupil: pupil data array 
        - pupil_size: pupil size in mm 
        - pupil_mask (ndarray): Boolean mask for pupil over the full slm.
        - small_pupil_mask (ndarray): Boolean mask for small square pupil on the slm.

    Returns:
    - summed_image: the summed image used for mask creation
    """

    # Use kwargs or default from dao_setup
    camera = kwargs.get("camera", dao_setup.camera_wfs)
    slm = kwargs.get("slm", dao_setup.slm)
    data_pupil = kwargs.get("data_pupil", dao_setup.data_pupil)
    pupil_size = kwargs.get("pupil_size", dao_setup.pupil_size)
    pupil_mask = kwargs.get("pupil_mask", dao_setup.pupil_mask)
    small_pupil_mask = kwargs.get("small_pupil_mask", dao_setup.small_pupil_mask)

    dataWidth = dao_setup.dataWidth
    dataHeight = dao_setup.dataHeight
    pixel_size_mm = dao_setup.pixel_size
    Npix = pupil_size / pixel_size_mm  # replaced npix_pupil with Npix

    modulation_img_arr = []

    # Setup dynamic plot if verbose_plot is True
    if verbose_plot:
        fig, ax = plt.subplots()
        img_display = ax.imshow(np.zeros((dataHeight, dataWidth)), cmap='inferno', norm=mcolors.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(img_display, ax=ax)
        ax.set_title("Captured Image at 0 Degrees")
        plt.ion()
        plt.show()

    for angle in modulation_angles:
        if verbose:
            print(f"Applying modulation angle: {angle}")
        
        modulation_step = apply_intensity_tilt(small_pupil_mask, Npix / 2, tilt_angle=angle)
        data_modulation = modulation_amp * modulation_step
        data_slm = compute_data_slm(data_dm=data_modulation)
        slm.set_data(data_slm)

        time.sleep(dao_setup.wait_time)  # wait for SLM update

        # Capture image using pylon SDK wrapper function
        img = camera.get_data()
        modulation_img_arr.append(img)

        if verbose_plot:
            img_display.set_data(img)
            img_display.set_clim(vmin=np.min(img), vmax=np.max(img))
            cbar.update_normal(img_display)
            ax.set_title(f'Captured Image at {angle} Degrees; max {np.max(img):.2f}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    # Turn off interactive plotting after loop
    if verbose_plot:
        plt.ioff()

    summed_image = np.sum(np.asarray(modulation_img_arr), axis=0)

    return summed_image


def create_flux_filtering_mask_trial(n_iter, verbose=False, verbose_plot=False, **kwargs):
    """
    Run push-pull image acquisition using random DM actuator patterns.

    Parameters:
    - n_iter: number of random actuator patterns to test
    - verbose: whether to print/display progress
    - verbose_plot: if True, live image display is updated
    - kwargs:
        - slm: SLM instance (default: dao_setup.slm)
        - camera: initialized camera object (default: dao_setup.camera_wfs)
        - deformable_mirror: DM object with actuator control (default: dao_setup.deformable_mirror)
        - npix_small_pupil_grid: resolution of the inner grid (default: dao_setup.npix_small_pupil_grid)

    Returns:
    - summed_image: result of push-pull image subtraction summed over all modes
    """

    slm = kwargs.get("slm", dao_setup.slm)
    camera = kwargs.get("camera", dao_setup.camera_wfs)
    deformable_mirror = kwargs.get("deformable_mirror", dao_setup.deformable_mirror)
    npix_small_pupil_grid = kwargs.get("npix_small_pupil_grid", dao_setup.npix_small_pupil_grid)

    dataWidth = dao_setup.dataWidth
    dataHeight = dao_setup.dataHeight

    nact_total = int(deformable_mirror.num_actuators)
    data_dm = np.zeros((npix_small_pupil_grid, npix_small_pupil_grid), dtype=np.float32)

    push_pull_pyr_img_arr = []

    if verbose_plot:
        fig, ax = plt.subplots()
        img_display = ax.imshow(np.zeros((dataHeight, dataWidth)), cmap='gray', vmin=0, vmax=255)
        cbar = plt.colorbar(img_display, ax=ax)
        ax.set_title("Captured Image")
        plt.ion()
        plt.show()

    for i in range(n_iter):
        if verbose:
            print(f"Iteration {i + 1}")

        act_random = np.random.choice([0, 1], size=nact_total)

        # Push
        deformable_mirror.actuators = act_random
        data_dm[:, :] = deformable_mirror.opd.shaped
        data_slm = compute_data_slm(data_dm=data_dm)
        slm.set_data(data_slm)
        time.sleep(dao_setup.wait_time)
        push_img = camera.get_data()

        # Pull
        deformable_mirror.actuators = -act_random
        data_dm[:, :] = deformable_mirror.opd.shaped
        data_slm = compute_data_slm(data_dm=data_dm)
        slm.set_data(data_slm)
        time.sleep(dao_setup.wait_time)
        pull_img = camera.get_data()

        push_pull_img = push_img - pull_img
        push_pull_pyr_img_arr.append(push_pull_img)

        if verbose_plot:
            img_display.set_data(push_pull_img)
            img_display.set_clim(vmin=push_pull_img.min(), vmax=push_pull_img.max())
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    if verbose_plot:
        plt.ioff()

    summed_image = np.sum(np.asarray(push_pull_pyr_img_arr), axis=0)

    return summed_image
