"""Detrending (flattening) utilities for TESS light curves.

Stellar variability — starspots, rotation, pulsations — produces
long-period flux modulations that can mask or mimic transits.
The ``flatten`` function removes these trends using a Savitzky-Golay
filter while preserving short-duration events like planetary transits.

Key parameter: ``window_length``
    The filter window in cadences.  A value of 1001 (~33 hours at 2-min
    cadence) is large enough to preserve transits of up to ~13 hours
    while removing variability on timescales of days.
"""

from lightkurve import LightCurve

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def flatten_lightcurve(
    light_curve: LightCurve,
    window_length: int = config.FLATTEN_WINDOW_LENGTH,
) -> LightCurve:
    """Remove long-period stellar variability via a Savitzky-Golay filter.

    Args:
        light_curve: Input light curve (should already be normalized).
        window_length: Width of the smoothing window in cadences.
            Must be an odd number.  Larger values preserve more of the
            transit signal but remove less variability.

    Returns:
        A flattened ``LightCurve`` where the out-of-transit baseline
        is approximately 1.0.
    """
    # lightkurve's flatten returns a tuple: (flattened_lc, trend_lc)
    flattened = light_curve.flatten(window_length=window_length)
    logger.debug("Flattened light curve with window_length=%d", window_length)
    return flattened
