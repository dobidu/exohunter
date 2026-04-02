"""Data cleaning utilities for TESS light curves.

Removes NaN values, outliers (cosmic rays, instrumental glitches),
and handles data gaps gracefully.
"""

from lightkurve import LightCurve

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def remove_nans(light_curve: LightCurve) -> LightCurve:
    """Remove cadences where flux or time is NaN.

    Args:
        light_curve: Input light curve (may contain NaN entries).

    Returns:
        A new ``LightCurve`` with NaN cadences removed.
    """
    n_before = len(light_curve)
    cleaned = light_curve.remove_nans()
    n_removed = n_before - len(cleaned)

    if n_removed > 0:
        logger.debug("Removed %d NaN cadences (%.1f%%)", n_removed, 100 * n_removed / n_before)

    return cleaned


def remove_outliers(
    light_curve: LightCurve,
    sigma: float = config.OUTLIER_SIGMA,
) -> LightCurve:
    """Remove flux outliers using sigma-clipping.

    Points that deviate more than ``sigma`` standard deviations from
    the local median are discarded.  This catches cosmic ray hits and
    instrumental glitches without affecting genuine transit dips (which
    are typically < 3σ for small planets).

    Args:
        light_curve: Input light curve.
        sigma: Clipping threshold in standard deviations.

    Returns:
        A new ``LightCurve`` with outliers removed.
    """
    n_before = len(light_curve)
    cleaned = light_curve.remove_outliers(sigma=sigma)
    n_removed = n_before - len(cleaned)

    if n_removed > 0:
        logger.debug(
            "Removed %d outliers at %.1fσ (%.1f%%)",
            n_removed,
            sigma,
            100 * n_removed / n_before,
        )

    return cleaned
