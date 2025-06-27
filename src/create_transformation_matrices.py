#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:40:10 2025

@author: laboptic
"""
import numpy as np
import scipy.linalg
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from hcipy import *
from src.kl_basis_eigenmodes import *
from src.utils import *
import sys
from pathlib import Path

# Configure root paths without changing the working directory
OPT_LAB_ROOT = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", OPT_LAB_ROOT / "PROJECTS_3/RISTRETTO/Banc AO"))
sys.path.append(str(OPT_LAB_ROOT))
sys.path.append(str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

# Output folders
folder_calib = ROOT_DIR / 'outputs/Calibration_files'
folder_pyr_mask = ROOT_DIR / 'outputs/3s_pyr_mask'
folder_transformation_matrices = ROOT_DIR / 'outputs/Transformation_matrices'
folder_closed_loop_tests = ROOT_DIR / 'outputs/Closed_loop_tests'
folder_turbulence = ROOT_DIR / 'outputs/Phase_screens'

# Default folder for storing transformation matrices

# Compute actuator-to-phase matrix
def compute_Act2Phs(nact, npix, IFs, folder, verbose=False):
    """
    Compute the actuator-to-phase transformation matrix (Act2Phs) and its inverse (Phs2Act).
    If the matrices exist as FITS files, they are loaded; otherwise, they are generated and saved.

    Parameters:
    nact (int): Number of actuators.
    npix (int): Number of pixels in the pupil grid.
    pupil_size (float): Physical size of the pupil.
    pupil_grid (array-like): The pupil grid.
    verbose (bool, optional): If True, prints debugging information.

    Returns:
    tuple:
        - Act2Phs (ndarray): Actuator-to-phase transformation matrix.
        - Phs2Act (ndarray): Inverse phase-to-actuator matrix.
    """
    act2phs_filename = f'Act2Phs_nact_{nact}_npupil_{npix}.fits'
    phs2act_filename = f'Phs2Act_npupil_{npix}_nact_{nact}.fits'
    
    if not matrix_exists(folder, act2phs_filename) or not matrix_exists(folder, phs2act_filename):
        Act2Phs = np.asarray(IFs)
        Phs2Act = scipy.linalg.pinv(Act2Phs)
        fits.writeto(os.path.join(folder, act2phs_filename), Act2Phs, overwrite=True)
        fits.writeto(os.path.join(folder, phs2act_filename), Phs2Act, overwrite=True)
        if verbose:
            print("Act2Phs and Phs2Act matrices created.")
    else:
        Act2Phs = fits.getdata(os.path.join(folder, act2phs_filename))
        Phs2Act = fits.getdata(os.path.join(folder, phs2act_filename))
        if verbose:
            print("Act2Phs and Phs2Act matrices loaded from file.")
    return Act2Phs, Phs2Act

# # Compute KL-to-phase matrix
# def compute_KL2Phs(nact, npix, nmodes_kl, IFs, pupil_mask, Act2Phs, folder, verbose=False):
#     """
#     Compute the KL-to-phase transformation matrix (KL2Phs) and its inverse (Phs2KL).
#     If matrices exist as FITS files, they are loaded; otherwise, they are generated and saved.

#     Parameters:
#     nact (int): Number of actuators.
#     npix (int): Number of pixels in the pupil grid.
#     pupil_size (float): Physical size of the pupil.
#     pupil_grid (array-like): The pupil grid.
#     pupil_mask (array-like): Mask applied to the pupil.
#     verbose (bool, optional): If True, prints debugging information.

#     Returns:
#     tuple:
#         - KL2Phs (ndarray): KL-to-phase transformation matrix.
#         - Phs2KL (ndarray): Inverse phase-to-KL matrix.
#     """

#     kl2phs_filename = f'KL2Phs_nkl_{nmodes_kl}_npupil_{npix}.fits'
#     phs2kl_filename = f'Phs2KL_npupil_{npix}_nkl_{nmodes_kl}.fits'
    
#     if not matrix_exists(folder, kl2phs_filename) or not matrix_exists(folder, phs2kl_filename):
#         IFs = IFs.transformation_matrix #.toarray()
#         IFs =  IFs * pupil_mask.flatten()[:, np.newaxis]
#         M2C, KLphiVec = computeEigenModes(IFs, pupil_mask.flatten())
#         KL2Act = M2C.T
#         KL2Phs = KL2Act @ Act2Phs
#         KL2Phs = np.asarray([mode/np.ptp(mode) for mode in KL2Phs[:nmodes_kl, :]])
#         print('KL2Phs', KL2Phs.shape)
#         plt.figure()
#         plt.imshow(KL2Phs[:, :200])
#         plt.show()
#         #KL2Phs = np.asarray([mode/np.ptp(mode) for mode in KLphiVec.T[:nmodes_kl, :]])
#         Phs2KL = scipy.linalg.pinv(KL2Phs)
#         fits.writeto(os.path.join(folder, kl2phs_filename), KL2Phs, overwrite=True)
#         fits.writeto(os.path.join(folder, phs2kl_filename), Phs2KL, overwrite=True)
#         if verbose:
#             print("KL2Phs and Phs2KL matrices created.")
#     else:
#         KL2Phs = fits.getdata(os.path.join(folder, kl2phs_filename))
#         Phs2KL = fits.getdata(os.path.join(folder, phs2kl_filename))
#         if verbose:
#             print("KL2Phs and Phs2KL matrices loaded from file.")
#     return KL2Phs, Phs2KL

# # Compute KL-to-actuator matrix
# def compute_KL2Act(nact, nmodes_kl, Act2Phs, Phs2Act, KL2Phs, Phs2KL, folder, verbose=False):
#     """
#     Compute the KL-to-actuator transformation matrix (KL2Act) and its inverse (Act2KL).
#     If matrices exist as FITS files, they are loaded; otherwise, they are computed and saved.

#     Parameters:
#     nact (int): Number of actuators.
#     Act2Phs (ndarray): Actuator-to-phase transformation matrix.
#     Phs2Act (ndarray): Phase-to-actuator matrix.
#     KL2Phs (ndarray): KL-to-phase transformation matrix.
#     Phs2KL (ndarray): Phase-to-KL matrix.
#     verbose (bool, optional): If True, prints debugging information.

#     Returns:
#     tuple:
#         - Act2KL (ndarray): Actuator-to-KL transformation matrix.
#         - KL2Act (ndarray): KL-to-actuator transformation matrix.
#     """

#     kl2act_filename = f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'
#     act2kl_filename = f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'
    
#     if not matrix_exists(folder, kl2act_filename) or not matrix_exists(folder, act2kl_filename):
#         Act2KL = Act2Phs @ Phs2KL
#         KL2Act = KL2Phs @ Phs2Act
#         fits.writeto(os.path.join(folder, kl2act_filename), KL2Act, overwrite=True)
#         fits.writeto(os.path.join(folder, act2kl_filename), Act2KL, overwrite=True)
#         if verbose:
#             print("KL2Act and Act2KL matrices created.")
#     else:
#         KL2Act = fits.getdata(os.path.join(folder, kl2act_filename))
#         Act2KL = fits.getdata(os.path.join(folder, act2kl_filename))
#         if verbose:
#             print("KL2Act and Act2KL matrices loaded from file.")
#     return Act2KL, KL2Act


# Compute KL-to-act matrix
def compute_KL2Act(nact, npix, nmodes_kl, IFs, pupil_mask, folder, verbose=False):
    """
    Compute the KL-to-act transformation matrix (KL2Act) and its inverse (Act2KL).
    If matrices exist as FITS files, they are loaded; otherwise, they are generated and saved.

    Parameters:
    nact (int): Number of actuators.
    npix (int): Number of pixels in the pupil grid.
    nmodes_kl (int): Number of KL modes.
    IFs (array-like): The dm influence functions.
    pupil_mask (array-like): Mask applied to the pupil.
    verbose (bool, optional): If True, prints debugging information.

    Returns:
    tuple:
        - Act2KL (ndarray): Actuator-to-KL transformation matrix.
        - KL2Act (ndarray): KL-to-actuator transformation matrix.
    """
    kl2act_filename = f'KL2Act_nkl_{nmodes_kl}_nact_{nact}.fits'
    act2kl_filename = f'Act2KL_nact_{nact}_nkl_{nmodes_kl}.fits'
    
    if not matrix_exists(folder, kl2act_filename) or not matrix_exists(folder, act2kl_filename):
        IFs = IFs.transformation_matrix #.toarray()
        IFs =  IFs * pupil_mask.flatten()[:, np.newaxis]
        M2C, KLphiVec = computeEigenModes(IFs, pupil_mask.flatten())
        KL2Act = np.asarray([mode for mode in M2C.T[:nmodes_kl, :]])
        Act2KL = scipy.linalg.pinv(KL2Act)
        KL2Phs = np.asarray([mode/np.ptp(mode) for mode in KLphiVec.T[:nmodes_kl, :]])
        fits.writeto(os.path.join(folder, kl2act_filename), KL2Act, overwrite=True)
        fits.writeto(os.path.join(folder, act2kl_filename), Act2KL, overwrite=True)
        if verbose:
            print("KL2Act and Act2KL matrices created.")
    else:
        KL2Act = fits.getdata(os.path.join(folder, kl2act_filename))
        Act2KL = fits.getdata(os.path.join(folder, act2kl_filename))
        if verbose:
            print("KL2Act and Act2KL matrices loaded from file.")
    return Act2KL, KL2Act

# Compute KL-to-phase matrix
def compute_KL2Phs(nact, npix, nmodes_kl, Act2Phs, Phs2Act, KL2Act, Act2KL, folder, verbose=False):
    """
    Compute the KL-to-actuator transformation matrix (KL2Act) and its inverse (Act2KL).
    If matrices exist as FITS files, they are loaded; otherwise, they are computed and saved.

    Parameters:
    nact (int): Number of actuators.
    npix (int): Number of pixels in the pupil grid.
    nmodes_kl (int): Number of KL modes.
    Act2Phs (ndarray): Actuator-to-phase transformation matrix.
    Phs2Act (ndarray): Phase-to-actuator matrix.
    KL2Phs (ndarray): KL-to-phase transformation matrix.
    Phs2KL (ndarray): Phase-to-KL matrix.
    verbose (bool, optional): If True, prints debugging information.

    Returns:
    tuple:
        - Act2KL (ndarray): Actuator-to-KL transformation matrix.
        - KL2Act (ndarray): KL-to-actuator transformation matrix.
    """

    kl2phs_filename = f'KL2Phs_nkl_{nmodes_kl}_npupil_{npix}.fits'
    phs2kl_filename = f'Phs2KL_npupil_{npix}_nkl_{nmodes_kl}.fits'
    
    if not matrix_exists(folder, kl2phs_filename) or not matrix_exists(folder, phs2kl_filename):
        KL2Phs = KL2Act @ Act2Phs
        KL2Phs = np.asarray([mode/np.ptp(mode) for mode in KL2Phs])
        Phs2KL = scipy.linalg.pinv(KL2Phs)
        fits.writeto(os.path.join(folder, kl2phs_filename), KL2Phs, overwrite=True)
        fits.writeto(os.path.join(folder, phs2kl_filename), Phs2KL, overwrite=True)
        if verbose:
            print("KL2Phs and Phs2KL matrices created.")
    else:
        KL2Phs = fits.getdata(os.path.join(folder, kl2phs_filename))
        Phs2KL = fits.getdata(os.path.join(folder, phs2kl_filename))
        if verbose:
            print("KL2Phs and Phs2KL matrices loaded from file.")
    return KL2Phs, Phs2KL

# Compute Zernike-to-phase matrix
def compute_Znk2Phs(nmodes_zernike, npix, pupil_size, pupil_grid, folder, verbose=False):
    """
    Compute the Zernike-to-phase transformation matrix (Znk2Phs) and its inverse (Phs2Znk).
    If matrices exist as FITS files, they are loaded; otherwise, they are computed and saved.

    Parameters:
    nmodes_zernike (int): Number of Zernike modes.
    npix (int): Number of pixels in the pupil grid.
    pupil_size (float): Physical size of the pupil.
    pupil_grid (array-like): The pupil grid.
    verbose (bool, optional): If True, prints debugging information.

    Returns:
    tuple:
        - Znk2Phs (ndarray): Zernike-to-phase transformation matrix.
        - Phs2Znk (ndarray): Phase-to-Zernike transformation matrix.
    """
    znk2phs_filename = f'Znk2Phs_nzernike_{nmodes_zernike}_npupil_{npix}.fits'
    phs2znk_filename = f'Phs2Znk_npupil_{npix}_nzernike_{nmodes_zernike}.fits'
    
    if not matrix_exists(folder, znk2phs_filename) or not matrix_exists(folder, phs2znk_filename):
        Znk2Phs = make_zernike_basis(nmodes_zernike, pupil_size, pupil_grid)
        Znk2Phs = np.asarray([mode/np.ptp(mode) for mode in Znk2Phs])
        Phs2Znk = scipy.linalg.pinv(Znk2Phs)
        fits.writeto(os.path.join(folder, znk2phs_filename), Znk2Phs, overwrite=True)
        fits.writeto(os.path.join(folder, phs2znk_filename), Phs2Znk, overwrite=True)
        if verbose:
            print("Znk2Phs and Phs2Znk matrices created.")
    else:
        Znk2Phs = fits.getdata(os.path.join(folder, znk2phs_filename))
        Phs2Znk = fits.getdata(os.path.join(folder, phs2znk_filename))
        if verbose:
            print("Znk2Phs and Phs2Znk matrices loaded from file.")
    return Znk2Phs, Phs2Znk

# Compute Zernike-to-actuator matrix
def compute_Znk2Act(nact, nmodes_zernike, Act2Phs, Phs2Act, Znk2Phs, Phs2Znk, folder, verbose=False):
    """
    Compute the Zernike-to-actuator transformation matrix (Znk2Act) and its inverse (Act2Znk).
    If matrices exist as FITS files, they are loaded; otherwise, they are computed and saved.

    Parameters:
    nact (int): Number of actuators.
    nmodes_zernike (int): Number of Zernike modes.
    Act2Phs (ndarray): Actuator-to-phase transformation matrix.
    Phs2Act (ndarray): Phase-to-actuator transformation matrix.
    Znk2Phs (ndarray): Zernike-to-phase transformation matrix.
    Phs2Znk (ndarray): Phase-to-Zernike transformation matrix.
    verbose (bool, optional): If True, prints debugging information.

    Returns:
    tuple:
        - Act2Znk (ndarray): Actuator-to-Zernike transformation matrix.
        - Znk2Act (ndarray): Zernike-to-actuator transformation matrix.
    """
    znk2act_filename = f'Znk2Act_nzernike_{nmodes_zernike}_nact_{nact}.fits'
    act2znk_filename = f'Act2Znk_nact_{nact}_nzernike_{nmodes_zernike}.fits'
    
    if not matrix_exists(folder, znk2act_filename) or not matrix_exists(folder, act2znk_filename):
        Act2Znk = Act2Phs @ Phs2Znk
        Znk2Act = Znk2Phs @ Phs2Act
        fits.writeto(os.path.join(folder, znk2act_filename), Znk2Act, overwrite=True)
        fits.writeto(os.path.join(folder, act2znk_filename), Act2Znk, overwrite=True)
        if verbose:
            print("Znk2Act and Act2Znk matrices created.")
    else:
        Znk2Act = fits.getdata(os.path.join(folder, znk2act_filename))
        Act2Znk = fits.getdata(os.path.join(folder, act2znk_filename))
        if verbose:
            print("Znk2Act and Act2Znk matrices loaded from file.")
    return Act2Znk, Znk2Act


# # Example usage

# nact = 10  # Example number of actuators
# npix_small_pupil_grid = 64  # Number of pixels in the small pupil grid
# pupil_size = 1.0  # Example pupil size (in arbitrary units)
# small_pupil_grid = np.linspace(-1, 1, npix_small_pupil_grid)  # Example pupil grid
# small_pupil_mask = np.ones((npix_small_pupil_grid, npix_small_pupil_grid))  # Example mask (full pupil)

# # Compute matrices
# Act2Phs, Phs2Act = compute_Act2Phs(nact, npix_small_pupil_grid, pupil_size, small_pupil_grid)
# KL2Phs, Phs2KL = compute_KL2Phs(nact, npix_small_pupil_grid, pupil_size, small_pupil_grid, small_pupil_mask)
# Act2KL, KL2Act = compute_KL2Act(nact, Act2Phs, Phs2Act, KL2Phs, Phs2KL)
