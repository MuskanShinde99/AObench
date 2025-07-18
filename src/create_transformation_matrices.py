#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:41:16 2025

@author: ristretto-dao
"""


# Import Libraries
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

from src.config import config

ROOT_DIR = config.root_dir

# Import Specific Modules
import dao
from src.dao_setup import init_setup
import src.dao_setup as dao_setup
from src.transformation_matrices_functions import *
from src.create_shared_memories import *

setup = init_setup()

Act2Phs, Phs2Act = compute_Act2Phs(
    setup.nact,
    setup.npix_small_pupil_grid,
    dao_setup.dm_modes_full,
    setup.folder_transformation_matrices,
    verbose=True,
)

# Create KL modes
nmodes_kl = dao_setup.nact_valid
# Act2KL, KL2Act = compute_KL2Act(nact, npix_small_pupil_grid, nmodes_kl, dm_modes_full, small_pupil_mask, folder_transformation_matrices, verbose=True)
# KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, nmodes_kl, Act2Phs, Phs2Act, KL2Act, Act2KL, folder_transformation_matrices, verbose=True)
KL2Act, KL2Phs = compute_KL(
    setup.nact,
    setup.npix_small_pupil_grid,
    nmodes_kl,
    dao_setup.dm_modes_full,
    Act2Phs,
    Phs2Act,
    dao_setup.small_pupil_mask,
    setup.folder_transformation_matrices,
    verbose=False,
)

print('KL2Act', KL2Act.dtype)


# set shared memories
KL2Act_shm.set_data(np.asarray(KL2Act))
KL2Phs_shm.set_data(np.asarray(KL2Phs))


# Plot KL projected| on actuators
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Act[mode].reshape(setup.nact, setup.nact), cmap='viridis')
    axes_flat[i].set_title(f' KL2Act {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

# Plot KL 
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array for easier indexing
axes_flat = axes.flatten()

for i, mode in enumerate(range(10)):
    im = axes_flat[i].imshow(KL2Phs[mode].reshape(setup.npix_small_pupil_grid, setup.npix_small_pupil_grid), cmap='viridis')
    axes_flat[i].set_title(f' KL2Phs {mode}')
    axes_flat[i].axis('off')
    fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

plt.tight_layout()
plt.show()

KL2Act = KL2Act_shm.get_data()
KL2Phs = KL2Phs_shm.get_data()

plt.figure()
plt.imshow(KL2Act[0,:].reshape(setup.nact,setup.nact))
plt.colorbar()
plt.title('KL mode')
plt.show()

plt.figure()
plt.imshow(setup.small_pupil_mask*KL2Phs[0,:].reshape(setup.npix_small_pupil_grid,setup.npix_small_pupil_grid))
plt.colorbar()
plt.title('KL mode')
plt.show()


# KL2Phs_new = KL2Act @  Act2Phs

# # Plot KL 
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# # Flatten the axes array for easier indexing
# axes_flat = axes.flatten()

# for i, mode in enumerate(range(10)):
#     im = axes_flat[i].imshow(
#         KL2Phs_new[mode].reshape(setup.npix_small_pupil_grid, setup.npix_small_pupil_grid)
#         * dao_setup.small_pupil_mask,
#         cmap='viridis',
#     )
#     axes_flat[i].set_title(f' KL2Phs new {mode}')
#     axes_flat[i].axis('off')
#     fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

# plt.tight_layout()
# plt.show()

# KL2Act_new = KL2Phs @  Phs2Act

# # Plot KL projected| on actuators
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# # Flatten the axes array for easier indexing
# axes_flat = axes.flatten()

# for i, mode in enumerate(range(10)):
#     im = axes_flat[i].imshow(KL2Act_new[mode].reshape(setup.nact, setup.nact), cmap='viridis')
#     axes_flat[i].set_title(f' KL2Act new {mode}')
#     axes_flat[i].axis('off')
#     fig.colorbar(im, ax=axes_flat[i], fraction=0.03, pad=0.04)

# plt.tight_layout()
# plt.show()

