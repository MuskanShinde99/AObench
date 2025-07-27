import time
import numpy as np

from src.dao_setup import init_setup, ROOT_DIR

setup = init_setup()
wait_time = setup.wait_time
pupil_setup = setup.pupil_setup
camera_fp = setup.camera_fp
camera_wfs = setup.camera_wfs
slm = setup.slm
from pathlib import Path
import re


def scan_othermode_amplitudes(test_values, mode_index, wait=wait_time,
                              update_setup_file=False):
    """Iterate over amplitude values for a specified othermode.

    The pupil configuration is restored to its original othermode amplitudes
    unless ``update_setup_file`` is ``True`` and an optimal value is stored.

    Parameters
    ----------
    test_values : sequence of float
        Amplitude values to test.
    mode_index : int
        Index in ``othermodes_amplitudes`` to update.
    wait : float, optional
        Delay between successive SLM updates. Defaults to ``wait_time``.
    update_setup_file : bool, optional
        If True, update ``othermodes_amplitudes`` in ``dao_setup.py``
        with the best-performing amplitude. If multiple amplitudes
        achieve the same maximum intensity, their mean value is stored.

    Returns
    -------
    float or None
        Amplitude yielding the best intensity, or ``None`` if no optimum was
        found.
    """

    original_amps = list(pupil_setup.othermodes_amplitudes)

    if mode_index < 0 or mode_index >= len(pupil_setup.othermodes_amplitudes):
        raise ValueError(
            f"mode_index must be between 0 and {len(pupil_setup.othermodes_amplitudes) - 1}"
        )

    best_amps = []
    best_intensity = -np.inf

    for amp in test_values:
        new_amps = list(pupil_setup.othermodes_amplitudes)
        new_amps[mode_index] = amp

        slm_data = pupil_setup.update_pupil(new_othermodes_amplitudes=new_amps)
        slm.set_data(slm_data)
        time.sleep(wait)

        # Capture focal-plane image and log stats
        # Capture and average 5 images
        num_images = 5
        images = [camera_fp.get_data() for i in range(num_images)]
        fp_img = np.mean(images, axis=0)
        intensity = np.max(fp_img)
        print(f"Amplitude {amp:.3f} -> max intensity {intensity:.3f}")

        if intensity > best_intensity:
            best_intensity = intensity
            best_amps = [amp]
        elif np.isclose(intensity, best_intensity, rtol=0, atol=1e-9):
            best_amps.append(amp)

    if best_amps:
        best_amp = float(np.mean(best_amps))
    else:
        best_amp = None

    print(f"Best amplitude {best_amp:.3f} with max intensity {best_intensity:.3f}")

    if update_setup_file and best_amp is not None:
        dao_setup_path = Path(ROOT_DIR) / 'src/dao_setup.py'
        with open(dao_setup_path, 'r') as file:
            content = file.read()

        match = re.search(r"othermodes_amplitudes\s*=\s*\[(.*?)\]", content)
        if match:
            values = [float(v.strip()) for v in match.group(1).split(',')]
            if mode_index >= len(values):
                raise ValueError('mode_index out of range in dao_setup.py')
            values[mode_index] = best_amp
            new_line = f"othermodes_amplitudes = {values}"
            updated_content = re.sub(r"othermodes_amplitudes\s*=\s*\[.*?\]", new_line, content)
            with open(dao_setup_path, 'w') as file:
                file.write(updated_content)
            print('Updated `othermodes_amplitudes` in dao_setup.py')
            pupil_setup.update_pupil(new_othermodes_amplitudes=values)
        else:
            print('Failed to update `othermodes_amplitudes` in dao_setup.py')
            pupil_setup.update_pupil(new_othermodes_amplitudes=original_amps)
    else:
        pupil_setup.update_pupil(new_othermodes_amplitudes=original_amps)

    return best_amp


def scan_othermode_amplitudes_wfs_std(test_values, mode_index, mask, wait=wait_time,
                                      update_setup_file=False):
    """Iterate over amplitude values for a specified othermode using WFS data.

    This variant captures images from the wavefront sensor camera and
    minimizes the standard deviation of valid pixels within the mask. The
    pupil configuration is restored to its original state unless
    ``update_setup_file`` is ``True`` and the best amplitude is recorded.

    Parameters
    ----------
    test_values : sequence of float
        Amplitude values to test.
    mode_index : int
        Index in ``othermodes_amplitudes`` to update.
    wait : float, optional
        Delay between successive SLM updates. Defaults to ``wait_time``.
    update_setup_file : bool, optional
        If True, update ``othermodes_amplitudes`` in ``dao_setup.py`` with
        the best-performing amplitude. If multiple amplitudes yield the same
        minimum standard deviation, their mean value is stored.

    Returns
    -------
    float or None
        Amplitude yielding the minimum standard deviation, or ``None`` if
        no optimum was found.
    """

    original_amps = list(pupil_setup.othermodes_amplitudes)

    if mode_index < 0 or mode_index >= len(pupil_setup.othermodes_amplitudes):
        raise ValueError(
            f"mode_index must be between 0 and {len(pupil_setup.othermodes_amplitudes) - 1}"
        )

    best_amps = []
    best_std = np.inf

    for amp in test_values:
        new_amps = list(pupil_setup.othermodes_amplitudes)
        new_amps[mode_index] = amp

        slm_data = pupil_setup.update_pupil(new_othermodes_amplitudes=new_amps)
        slm.set_data(slm_data)
        time.sleep(wait)

        # Capture WFS images and compute standard deviation over valid pixels
        num_images = 5
        images = [camera_wfs.get_data() for _ in range(num_images)]
        wfs_img = np.mean(images, axis=0)
        pixel_std = np.std(wfs_img[mask])
        print(f"Amplitude {amp:.3f} -> std of valid pixels {pixel_std:.3f}")

        if pixel_std < best_std:
            best_std = pixel_std
            best_amps = [amp]
        elif np.isclose(pixel_std, best_std, rtol=0, atol=1e-9):
            best_amps.append(amp)

    if best_amps:
        best_amp = float(np.mean(best_amps))
    else:
        best_amp = None

    print(f"Best amplitude {best_amp:.3f} with min std {best_std:.3f}")

    if update_setup_file and best_amp is not None:
        dao_setup_path = Path(ROOT_DIR) / 'src/dao_setup.py'
        with open(dao_setup_path, 'r') as file:
            content = file.read()

        match = re.search(r"othermodes_amplitudes\s*=\s*\[(.*?)\]", content)
        if match:
            values = [float(v.strip()) for v in match.group(1).split(',')]
            if mode_index >= len(values):
                raise ValueError('mode_index out of range in dao_setup.py')
            values[mode_index] = best_amp
            new_line = f"othermodes_amplitudes = {values}"
            updated_content = re.sub(r"othermodes_amplitudes\s*=\s*\[.*?\]", new_line, content)
            with open(dao_setup_path, 'w') as file:
                file.write(updated_content)
            print('Updated `othermodes_amplitudes` in dao_setup.py')
            pupil_setup.update_pupil(new_othermodes_amplitudes=values)
        else:
            print('Failed to update `othermodes_amplitudes` in dao_setup.py')
            pupil_setup.update_pupil(new_othermodes_amplitudes=original_amps)
    else:
        pupil_setup.update_pupil(new_othermodes_amplitudes=original_amps)

    return best_amp
