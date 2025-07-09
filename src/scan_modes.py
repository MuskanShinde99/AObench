"""Utility functions for scanning SLM modes."""

import time
import numpy as np

from src.dao_setup import *
from src.utils import compute_data_slm


def scan_othermode_amplitudes(test_values, mode_index, wait=wait_time):
    """Iterate over amplitude values for a specified othermode.

    Parameters
    ----------
    test_values : sequence of float
        Amplitude values to test.
    mode_index : int
        Index in ``othermodes_amplitudes`` to update.
    wait : float, optional
        Delay between successive SLM updates. Defaults to ``wait_time``.
    """

    if mode_index < 0 or mode_index >= len(othermodes_amplitudes):
        raise ValueError(
            f"mode_index must be between 0 and {len(othermodes_amplitudes) - 1}"
        )

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
