import time
import numpy as np

from src.dao_setup import (
    wait_time,
    othermodes_amplitudes,
    update_pupil,
    pupil_mask,
    small_pupil_mask,
    offset_height,
    offset_width,
    npix_small_pupil_grid,
    camera_fp,
    slm,
    ROOT_DIR,
)
from src.utils import compute_data_slm
from pathlib import Path
import re


def scan_othermode_amplitudes(test_values, mode_index, wait=wait_time,
                              update_setup_file=False):
    """Iterate over amplitude values for a specified othermode.

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
    """

    if mode_index < 0 or mode_index >= len(othermodes_amplitudes):
        raise ValueError(
            f"mode_index must be between 0 and {len(othermodes_amplitudes) - 1}"
        )

    best_amps = []
    best_intensity = -np.inf

    for amp in test_values:
        new_amps = list(othermodes_amplitudes)
        new_amps[mode_index] = amp

        pupil = update_pupil(new_othermodes_amplitudes=new_amps)

        pupil_outer = np.copy(pupil)
        pupil_outer[pupil_mask] = 0

        pupil_inner = np.copy(
            pupil[offset_height:offset_height + npix_small_pupil_grid,
                  offset_width:offset_width + npix_small_pupil_grid]
        )
        pupil_inner[~small_pupil_mask] = 0

        slm_data = compute_data_slm(
            data_pupil_inner=pupil_inner,
            data_pupil_outer=pupil_outer,
            pupil_mask=pupil_mask,
            small_pupil_mask=small_pupil_mask,
        )
        slm.set_data(slm_data)
        time.sleep(wait)

        # Capture focal-plane image and log stats
        fp_img = camera_fp.get_data()
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
        else:
            print('Failed to update `othermodes_amplitudes` in dao_setup.py')