"""Normalization utilities for TESS light curves.

Normalizes flux values so that the median flux equals 1.0, which is
the standard convention for transit detection (a transit of depth 0.01
means the star dims by 1%).
"""

from lightkurve import LightCurve

from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_lightcurve(light_curve: LightCurve) -> LightCurve:
    """Normalize a light curve to median flux = 1.0.

    Args:
        light_curve: Input light curve with arbitrary flux units.

    Returns:
        A new ``LightCurve`` with flux divided by its median value.
    """
    normalized = light_curve.normalize()
    logger.debug("Normalized light curve (median flux → 1.0)")
    return normalized
