"""Local cache for TESS light curves.

Avoids redundant downloads by serializing ``LightCurve`` objects to FITS
files on disk.  The cache key is the TIC ID, sanitised to a safe filename.

Cache layout::

    data/cache/
        TIC_150428135.fits
        TIC_261136679.fits
        ...
"""

import re
from pathlib import Path

from lightkurve import LightCurve

from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def _tic_to_filename(tic_id: str) -> str:
    """Convert a TIC ID string into a safe filename.

    Args:
        tic_id: e.g. ``"TIC 150428135"``

    Returns:
        A filename like ``"TIC_150428135.fits"``.
    """
    # Keep only alphanumeric characters and replace spaces with underscores
    safe = re.sub(r"[^A-Za-z0-9]+", "_", tic_id.strip())
    return f"{safe}.fits"


def load_from_cache(tic_id: str, cache_dir: Path) -> LightCurve | None:
    """Attempt to load a cached light curve from disk.

    Args:
        tic_id: Target identifier.
        cache_dir: Directory where cache files are stored.

    Returns:
        A ``LightCurve`` object if a cache file exists, otherwise ``None``.
    """
    path = cache_dir / _tic_to_filename(tic_id)
    if not path.exists():
        return None

    try:
        light_curve = LightCurve.read(path, format="fits")
        logger.debug("Loaded cached light curve from %s", path)
        return light_curve
    except Exception:
        logger.warning("Failed to read cache file %s — will re-download", path)
        return None


def save_to_cache(light_curve: LightCurve, tic_id: str, cache_dir: Path) -> Path:
    """Persist a light curve to the local cache.

    Args:
        light_curve: The ``LightCurve`` to save.
        tic_id: Target identifier (used as the cache key).
        cache_dir: Directory where cache files are stored.

    Returns:
        The ``Path`` to the written FITS file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _tic_to_filename(tic_id)

    try:
        light_curve.to_fits(path, overwrite=True)
        logger.debug("Cached light curve to %s", path)
    except Exception:
        logger.warning("Failed to write cache file %s", path, exc_info=True)

    return path
