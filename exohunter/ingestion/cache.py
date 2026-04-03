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

import numpy as np
from astropy.table import Table
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

    Uses astropy Table I/O to read the FITS file and reconstruct a
    ``LightCurve`` from the ``time`` and ``flux`` columns.  This is
    more robust than ``LightCurve.read()`` which can fail on FITS
    files written by ``to_fits()`` due to non-standard time column
    metadata.

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
        table = Table.read(path, format="fits")

        time_col = table["time"].data.astype(np.float64)
        flux_col = table["flux"].data.astype(np.float64)

        flux_err_col = None
        if "flux_err" in table.colnames:
            flux_err_col = table["flux_err"].data.astype(np.float64)

        lc = LightCurve(time=time_col, flux=flux_col, flux_err=flux_err_col)
        logger.debug("Cache hit: loaded %d cadences from %s", len(lc), path)
        return lc

    except Exception:
        logger.warning("Failed to read cache file %s — will re-download", path)
        return None


def save_to_cache(light_curve: LightCurve, tic_id: str, cache_dir: Path) -> Path:
    """Persist a light curve to the local cache.

    Saves as a simple FITS table with ``time``, ``flux``, and
    ``flux_err`` columns using astropy's Table writer for reliable
    roundtrip I/O.

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
        # Build a plain astropy Table — avoids the non-standard time
        # metadata that makes LightCurve.read() fail on round-trip.
        columns = {
            "time": np.array(light_curve.time.value, dtype=np.float64),
            "flux": np.array(light_curve.flux.value, dtype=np.float64),
        }
        if light_curve.flux_err is not None:
            columns["flux_err"] = np.array(light_curve.flux_err.value, dtype=np.float64)

        table = Table(columns)
        table.write(path, format="fits", overwrite=True)
        logger.debug("Cached %d cadences to %s", len(light_curve), path)

    except Exception:
        logger.warning("Failed to write cache file %s", path, exc_info=True)

    return path
