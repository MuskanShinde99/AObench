#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAO calibrations — CLI-selectable cells

Run whole file from terminal and choose which cells execute.

Usage examples:
  # Run everything (default)
  python dao_calibrations_file.py

  # Run only mask creation and centering
  python dao_calibrations_file.py --cells create_mask,center_psf

  # Run all except the terminal DM reset
  python dao_calibrations_file.py --skip capture_reference,calibration_push_pull

  # See available cells
  python dao_calibrations_file.py --list

Notes:
- Each former "#%%" notebook cell is guarded by a boolean flag derived from --cells/--skip.
- Cells later in the pipeline try to fall back to shared memories if earlier cells were skipped.
- Plotting can be disabled with --no-plots.
"""

from __future__ import annotations

# Standard library
import os
import sys
import argparse
from datetime import datetime
import time

# Third-party
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# Project imports
import dao
from src.dao_setup import init_setup, las
from src.utils import set_data_dm, reload_setup
from src.config import config
from src.utils import *  # noqa: F401,F403 (kept for compatibility)
from src.circular_pupil_functions import *  # noqa
from src.flux_filtering_mask_functions import *  # noqa
from src.tilt_functions import *  # noqa
from src.calibration_functions import *  # noqa
from src.kl_basis_eigenmodes_functions import computeEigenModes, computeEigenModes_notsquarepupil  # noqa
from src.transformation_matrices_functions import *  # noqa
from src.psf_centring_algorithm_functions import *  # noqa
from src.shm_loader import shm
from src.scan_modes_functions import *  # noqa
from src.ao_loop_functions import *  # noqa

# ------------------------- CLI wiring -------------------------
CELL_ORDER = [
    "setup",
    "capture_bias",
    "set_dm_flat",
    "load_transformation_matrices",
    "create_mask",
    "center_psf",
    "scan_modes",
    "capture_reference",
    "calibration_push_pull",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run DAO calibration pipeline with selectable cells.")
    p.add_argument(
        "--cells",
        type=str,
        default="all",
        help="Comma-separated list of cells to run (or 'all'). Available: " + ",".join(CELL_ORDER),
    )
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of cells to skip.",
    )
    p.add_argument("--list", action="store_true", help="List cells and exit.")
    p.add_argument("--no-plots", action="store_true", help="Disable matplotlib windows.")

    # Bias capture options
    p.add_argument("--n-frames-bias", type=int, default=1000, help="Frames for bias image (median)")
    p.add_argument("--laser-wait", type=float, default=2.0, help="Seconds to wait after laser toggle")

    # Create mask options
    p.add_argument("--flux-cutoff", type=float, default=0.08,
                   help="Flux cutoff for mask (default: 0.08)")
    p.add_argument("--mod-amp", type=float, default=2.0, help="Modulation amplitude (λ/D)")
    p.add_argument("--mod-steps", type=int, default=2000, help="Number of modulation angles")
    p.add_argument("--dm-random-iters", type=int, default=500, help="DM random iterations")
    p.add_argument("--on-sky", "--onsky", dest="on_sky", action="store_true", 
                   help="Use on-sky acquisition for mask creation (default: off ).")

    # center_psf options
    p.add_argument("--center-bounds", type=str, default="-2,2,-2,2",
                   help="Centering bounds as xmin,xmax,ymin,ymax (default: -2,2,-2,2)")
    p.add_argument("--center-var-thresh", type=float, default=0.1,
                   help="Centering variance threshold (default: 0.1)")
    
    # scan_modes options
    p.add_argument("--scan-start", type=float, default=-0.5, help="Scan start")
    p.add_argument("--scan-stop", type=float, default=0.5, help="Scan stop")
    p.add_argument("--scan-step", type=float, default=0.05, help="Scan step")
    p.add_argument("--scan-mode-index", type=int, default=3, help="Mode index for scanning")

    # capture_reference options
    p.add_argument("--n-frames-ref-wfs", type=int, default=1000, help="Frames for WFS reference image")
    p.add_argument("--n-frames-ref-fp", type=int, default=1000, help="Frames for FP image")

    # calibration_push_pull options
    p.add_argument("--phase-amp", type=float, default=0.05, help="Phase amplitude for calibration")
    p.add_argument("--cal-reps", type=int, default=1, help="Calibration repetitions")
    p.add_argument("--mode-reps", type=str, default="2,2",
                   help="Mode repetitions as list (e.g. '2,2' or single integer '200' to repeat all)")
    p.add_argument("--n-frames-cal", type=int, default=100, help="Frames per mode during calibration")

    

    # ---------- create_summed_image toggle (default True) ----------
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--create-summed-image",
        dest="create_summed_image",
        action="store_true",
        help="Create summed image during mask creation (default: enabled)."
    )
    group.add_argument(
        "--no-summed-image",
        dest="create_summed_image",
        action="store_false",
        help="Disable summed image creation during mask creation."
    )
    p.set_defaults(create_summed_image=True)

    return p


# ------------------------- Helpers -------------------------

def parse_bounds(s: str):
    try:
        xmin, xmax, ymin, ymax = map(float, s.split(","))
        return [(xmin, xmax), (ymin, ymax)]
    except Exception:
        raise ValueError("--center-bounds must be 'xmin,xmax,ymin,ymax'")


def parse_mode_reps(s: str, nmodes_kl: int) -> list[int]:
    s = s.strip()
    if "," in s:
        parts = [int(x) for x in s.split(",") if x]
        return parts
    else:
        # single int => repeat first len(parts) modes that many times or all modes
        n = int(s)
        return [n] * nmodes_kl


def setup_matplotlib(disable: bool):
    if disable:
        plt.ioff()
    else:
        plt.ion()


# ------------------------- Main pipeline -------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available cells in order:")
        for i, name in enumerate(CELL_ORDER, 1):
            print(f" {i:>2}. {name}")
        sys.exit(0)

    # Decide which cells to run
    run_flags = {name: False for name in CELL_ORDER}
    if args.cells.lower() == "all":
        for k in run_flags:
            run_flags[k] = True
    else:
        requested = {s.strip() for s in args.cells.split(",") if s.strip()}
        unknown = requested - set(CELL_ORDER)
        if unknown:
            print("Unknown cell names:", ", ".join(sorted(unknown)))
            sys.exit(1)
        for k in requested:
            run_flags[k] = True

    if args.skip:
        for s in args.skip.split(","):
            s = s.strip()
            if s:
                if s not in run_flags:
                    print(f"Warning: skip name '{s}' not in known cells")
                run_flags[s] = False

    # --- Always run the setup cell ---
    # Force-enable setup regardless of --cells or --skip
    run_flags["setup"] = True

    setup_matplotlib(args.no_plots)

    # Top-level: echo global flags
    print("[global] cells =", args.cells)
    print("[global] skip =", args.skip if args.skip else "(none)")
    print("[global] no_plots =", args.no_plots)

    # Shared context across cells
    folder_calib = config.folder_calib
    setup = None
    KL2Act_papy = None
    mask = None
    reference_image = None


    # -------- Cell: setup --------
    if run_flags["setup"]:
        print("[setup] Initializing setup and shared memories…")
        # (no flags specific to setup cell)
        setup = init_setup()
        setup = reload_setup()

        # Bind shared memories (aliases)
        bias_image_shm = shm.bias_image_shm
        valid_pixels_mask_shm = shm.valid_pixels_mask_shm
        npix_valid_shm = shm.npix_valid_shm
        reference_image_shm = shm.reference_image_shm
        normalized_ref_image_shm = shm.normalized_ref_image_shm
        reference_psf_shm = shm.reference_psf_shm
        KL2Act_papy_shm = shm.KL2Act_papy_shm
        dm_flat_papy_shm = shm.dm_flat_papy_shm
        dm_papy_shm = shm.dm_papy_shm
        
        
    # # -------- Cell: capture_bias --------
    
    # # If capture_bias is skipped, attempt to auto-load bias image from disk
    # if not run_flags.get("capture_bias", False):
    #     try:
    #         bias_path = folder_calib / 'bias_image.fits'
    #         if hasattr(bias_path, 'exists') and bias_path.exists():
    #             bias_image = fits.getdata(bias_path)
    #         else:
    #             bias_image = fits.getdata(os.path.join(folder_calib, 'bias_image.fits'))
    #         shm.bias_image_shm.set_data(bias_image)
    #         print("[capture_bias] skipped -> loaded bias_image from calib folder (bias_image.fits)")
    #     except Exception:
    #         print("[auto-load] bias_image.fits not found.")
            
    # # If capture_bias is done
    # if run_flags["capture_bias"]:
    #     print("[capture_bias] Capturing bias image…")
    #     print(f"[capture_bias] n_frames_bias = {args.n_frames_bias}")
    #     print(f"[capture_bias] laser_wait = {args.laser_wait} s")
    #     if setup is None:
    #         setup = init_setup()
    #         setup = reload_setup()
    #     # Laser OFF
    #     try:
    #         if las is not None:
    #             las.enable(0)
    #             time.sleep(args.laser_wait)
    #             print("The laser is OFF")
    #         else:
    #             input("Turn OFF the laser and press Enter to continue")
    #     except Exception as e:
    #         print(f"[capture_bias] Laser OFF control failed: {e}. Prompting manual switch.")
    #         input("Turn OFF the laser and press Enter to continue")
    #     # ... (rest unchanged)


    # -------- Cell: set_dm_flat --------
    if run_flags["set_dm_flat"]:
        print("[set_dm_flat] Setting DM to flat…")
        # (no tunable flags here)
        if setup is None:
            setup = init_setup()
            setup = reload_setup()
        set_data_dm(setup=setup)
        # Optionally write flat to disk
        # fits.writeto(folder_calib / 'dm_flat_papy.fits', setup.dm_flat.astype(np.float32), overwrite=True)


    # -------- Cell: load_transformation_matrices --------
    if run_flags["load_transformation_matrices"]:
        print("[load_transformation_matrices] Loading KL2Act from shared memory…")
        # (no tunable flags here)
        try:
            KL2Act_papy = shm.KL2Act_papy_shm.get_data().T
            print(f"[load_transformation_matrices] KL2Act_papy shape = {KL2Act_papy.shape}")
        except Exception as e:
            print("ERROR: Could not read KL2Act_papy from shared memory:", e)
            sys.exit(2)


    # -------- Cell: create_mask --------
    if run_flags["create_mask"]:
        print("[create_mask] Creating flux filtering mask…")
        if KL2Act_papy is None:
            KL2Act_papy = shm.KL2Act_papy_shm.get_data().T
        method = 'dm_random'
        flux_cutoff = args.flux_cutoff
        modulation_angles = np.arange(0, args.mod_steps, 1)
        modulation_amp = args.mod_amp
        n_iter = args.dm_random_iters

        # New: report chosen flags for this cell
        print(f"[create_mask] flux_cutoff = {flux_cutoff}")
        print(f"[create_mask] mod_amp = {modulation_amp} (λ/D)")
        print(f"[create_mask] mod_steps = {args.mod_steps}")
        print(f"[create_mask] dm_random_iters = {n_iter}")
        print(f"[create_mask] OnSky = {args.on_sky}")
        print(f"[create_mask] create_summed_image = {args.create_summed_image}")

        mask = create_flux_filtering_mask(
            method, flux_cutoff, KL2Act_papy[0], KL2Act_papy[1],
            modulation_angles, modulation_amp, n_iter,
            create_summed_image=args.create_summed_image,
            verbose=False, verbose_plot=True,
            OnSky=args.on_sky,
        )
        print(f"[create_mask] Mask dimensions: {mask.shape}")
        shm.valid_pixels_mask_shm.set_data(mask)

        # compute npix_valid
        valid_pixels_indices = np.where(mask > 0)
        npix_valid = valid_pixels_indices[0].shape[0]
        shm.npix_valid_shm.set_data(np.array([[npix_valid]]))
        print(f"[create_mask] Number of valid pixels = {npix_valid}")

        # Reset DM to flat after mask creation
        set_data_dm(setup=setup)

        # Create dependent shared memories
        KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm',
                                   np.zeros((setup.nmodes_KL, setup.img_size_wfs_cam_x*setup.img_size_wfs_cam_y), dtype=np.float64))
        slopes_shm = dao.shm('/tmp/slopes.im.shm', np.zeros((npix_valid, 1), dtype=np.uint64))
        KL2S_shm = dao.shm('/tmp/KL2S.im.shm', np.zeros((setup.nmodes_KL, npix_valid), dtype=np.float64))
        S2KL_shm = dao.shm('/tmp/S2KL.im.shm', np.zeros((npix_valid, setup.nmodes_KL), dtype=np.float64))


    # # -------- Cell: center_psf --------
    
    # if run_flags["center_psf"]:
    #     print("[center_psf] Centering PSF on pyramid tip…")
    #     print(f"[center_psf] center_bounds = {args.center_bounds}")
    #     print(f"[center_psf] center_var_thresh = {args.center_var_thresh}")
    #     if mask is None:
    #         try:
    #             mask = shm.valid_pixels_mask_shm.get_data()
    #         except Exception:
    #             print("ERROR: valid mask not available. Run 'create_mask' first or load it into shared memory.")
    #             sys.exit(3)
    #     bounds = parse_bounds(args.center_bounds)
    #     center_psf_on_pyramid_tip(
    #         mask=mask,
    #         bounds=bounds,
    #         variance_threshold=args.center_var_thresh,
    #         update_setup_file=True,
    #         verbose=True,
    #         verbose_plot=False,
    #     )


    # # -------- Cell: scan_modes --------
    
    # if run_flags["scan_modes"]:
    #     print("[scan_modes] Scanning mode amplitudes…")
    #     print(f"[scan_modes] scan_start = {args.scan_start}")
    #     print(f"[scan_modes] scan_stop = {args.scan_stop}")
    #     print(f"[scan_modes] scan_step = {args.scan_step}")
    #     print(f"[scan_modes] scan_mode_index = {args.scan_mode_index}")
    #     if mask is None:
    #         try:
    #             mask = shm.valid_pixels_mask_shm.get_data()
    #         except Exception:
    #             print("ERROR: valid mask not available. Run 'create_mask' first or load it into shared memory.")
    #             sys.exit(4)
    #     test_values = np.arange(args.scan_start, args.scan_stop, args.scan_step)
    #     mode_index = args.scan_mode_index
    #     scan_othermode_amplitudes_wfs_std(
    #         test_values, mode_index, mask, update_setup_file=False
    #     )


    # -------- Cell: capture_reference --------
    
    # If capture_reference is skipped, attempt to auto-load reference images from disk
    if not run_flags.get("capture_reference", False):
        print("[capture_reference] skipped -> will try auto-load from disk.")
        try:
            ref_path = folder_calib / 'reference_image_raw.fits'
            if hasattr(ref_path, 'exists') and ref_path.exists():
                reference_image = fits.getdata(ref_path)
                shm.reference_image_shm.set_data(reference_image)
                print(f"[capture_reference] auto-loaded reference_image ({ref_path.name})")
            else:
                try:
                    reference_image = fits.getdata(os.path.join(folder_calib, 'reference_image_raw.fits'))
                    shm.reference_image_shm.set_data(reference_image)
                    print("[capture_reference] auto-loaded reference_image (reference_image_raw.fits)")
                except Exception:
                    print("[capture_reference] auto-load: reference_image_raw.fits not found; will rely on shared memory or fail later.")

            psf_path = folder_calib / 'reference_psf.fits'
            if hasattr(psf_path, 'exists') and psf_path.exists():
                fp_image = fits.getdata(psf_path)
                shm.reference_psf_shm.set_data(fp_image)
                print(f"[capture_reference] auto-loaded reference_psf ({psf_path.name})")
            else:
                try:
                    fp_image = fits.getdata(os.path.join(folder_calib, 'reference_psf.fits'))
                    shm.reference_psf_shm.set_data(fp_image)
                    print("[capture_reference] auto-loaded reference_psf (reference_psf.fits)")
                except Exception:
                    print("[capture_reference] auto-load: reference_psf.fits not found.")
        except Exception as e:
            print("[capture_reference] auto-load error from folder_calib:", e)

    # If capture_reference is done
    if run_flags["capture_reference"]:
        print("[capture_reference] Capturing reference images…")
        print(f"[capture_reference] n_frames_ref_wfs = {args.n_frames_ref_wfs}")
        print(f"[capture_reference] n_frames_ref_fp = {args.n_frames_ref_fp}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # (Re)load setup to access cameras
        if setup is None:
            setup = init_setup()
            setup = reload_setup()

        try:
            camera_wfs = setup.camera_wfs
            camera_fp = setup.camera_fp
        except Exception as e:
            print("ERROR: Cameras not available from setup:", e)
            sys.exit(5)

        # WFS reference image
        n_frames = args.n_frames_ref_wfs
        reference_image = (
            np.mean([camera_wfs.get_data().astype(np.float32) for _ in range(n_frames)], axis=0)
        ).astype(camera_wfs.get_data().dtype)
        shm.reference_image_shm.set_data(reference_image)
        fits.writeto(folder_calib / 'reference_image_raw.fits', reference_image, overwrite=True)
        fits.writeto(folder_calib / f'reference_image_raw_{timestamp}.fits', reference_image, overwrite=True)

        if not args.no_plots:
            plt.figure(); plt.imshow(reference_image); plt.colorbar(); plt.title('Reference Image'); plt.show()

        # Focal plane PSF image
        n_frames_fp = args.n_frames_ref_fp
        fp_image = (
            np.mean([camera_fp.get_data().astype(np.float32) for _ in range(n_frames_fp)], axis=0)
        ).astype(camera_fp.get_data().dtype)
        shm.reference_psf_shm.set_data(fp_image)
        fits.writeto(folder_calib / 'reference_psf.fits', fp_image, overwrite=True)
        fits.writeto(folder_calib / f'reference_psf_{timestamp}.fits', fp_image, overwrite=True)

        if not args.no_plots:
            plt.figure(); plt.imshow(fp_image); plt.colorbar(); plt.title('PSF'); plt.show()
            plt.figure(); plt.plot(fp_image[:, 253:273]); plt.title('PSF radial profile'); plt.show()

    # -------- Cell: calibration_push_pull --------
    if run_flags["calibration_push_pull"]:
        print("[calibration_push_pull] Running push/pull calibration…")
        print(f"[calibration_push_pull] phase_amp = {args.phase_amp}")
        print(f"[calibration_push_pull] cal_reps = {args.cal_reps}")
        print(f"[calibration_push_pull] mode_reps (raw) = {args.mode_reps}")
        print(f"[calibration_push_pull] n_frames_cal = {args.n_frames_cal}")
        if KL2Act_papy is None:
            KL2Act_papy = shm.KL2Act_papy_shm.get_data().T
        if reference_image is None:
            try:
                reference_image = shm.reference_image_shm.get_data()
            except Exception:
                # Fallback to disk if capture_reference was skipped
                try:
                    reference_image = fits.getdata(folder_calib / 'reference_image_raw.fits')
                    shm.reference_image_shm.set_data(reference_image)
                    print("[calibration_push_pull] Loaded reference_image from folder_calib/reference_image_raw.fits")
                except Exception:
                    try:
                        reference_image = fits.getdata(os.path.join(folder_calib, 'reference_image_raw.fits'))
                        shm.reference_image_shm.set_data(reference_image)
                        print("[calibration_push_pull] Loaded reference_image from folder_calib/reference_image_raw.fits")
                    except Exception:
                        print("ERROR: reference_image not available. Provide reference_image_raw.fits in folder_calib or run 'capture_reference'.")
                        sys.exit(6)

        phase_amp = args.phase_amp
        calibration_repetitions = args.cal_reps
        mode_repetitions = parse_mode_reps(args.mode_reps, setup.nmodes_KL)
        print(f"[calibration_push_pull] mode_reps (parsed) length = {len(mode_repetitions)}")

        response_matrix_full, response_matrix_filtered = create_response_matrix(
            KL2Act_papy,
            phase_amp,
            reference_image,
            mask if mask is not None else shm.valid_pixels_mask_shm.get_data(),
            verbose=True,
            verbose_plot=False,
            calibration_repetitions=calibration_repetitions,
            mode_repetitions=mode_repetitions,
            push_pull=False,
            pull_push=True,
            n_frames=args.n_frames_cal,
        )

        # Reset DM to flat
        set_data_dm(setup=setup)

        print("[calibration_push_pull] Full response matrix shape:", response_matrix_full.shape)
        print("[calibration_push_pull] Filtered response matrix shape:", response_matrix_filtered.shape)

        if not args.no_plots:
            plt.figure(); plt.imshow(response_matrix_filtered, cmap='gray', aspect='auto')
            plt.title('Filtered Push-Pull Response Matrix'); plt.xlabel('Slopes'); plt.ylabel('Modes'); plt.colorbar(); plt.show()

        # Save into shared memories
        KL2PWFS_cube_shm = dao.shm('/tmp/KL2PWFS_cube.im.shm',
                                   np.zeros((setup.nmodes_KL, setup.img_size_wfs_cam_x*setup.img_size_wfs_cam_y), dtype=np.float64))
        KL2PWFS_cube_shm.set_data(np.asanyarray(response_matrix_full).astype(np.float64))

        KL2S_shm = dao.shm('/tmp/KL2S.im.shm', np.zeros((setup.nmodes_KL, response_matrix_filtered.shape[1]), dtype=np.float64))
        KL2S_shm.set_data(np.asanyarray(response_matrix_filtered).astype(np.float64))

    print("Done.")
    
    
    if not args.no_plots:
        plt.ioff()            # ensure blocking mode
        plt.show()            # block here until windows are closed

