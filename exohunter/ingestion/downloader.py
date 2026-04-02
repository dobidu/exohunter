"""Concurrent downloader for TESS light curves via lightkurve.

This module handles all network interaction with MAST (Mikulski Archive
for Space Telescopes).  Downloads are I/O-bound, so we use a
``ThreadPoolExecutor`` to fetch multiple targets in parallel.

Example::

    from exohunter.ingestion.downloader import download_batch

    light_curves = download_batch(["TIC 150428135", "TIC 261136679"])
"""

import time
from pathlib import Path

import lightkurve as lk
from lightkurve import LightCurve

from exohunter import config
from exohunter.ingestion.cache import load_from_cache, save_to_cache
from exohunter.utils.logging import get_logger
from exohunter.utils.parallel import run_parallel_threads
from exohunter.utils.timing import timing

logger = get_logger(__name__)


def search_targets(
    sector: int,
    author: str = "SPOC",
    limit: int | None = None,
) -> list[str]:
    """Search for TIC IDs observed in a given TESS sector.

    Args:
        sector: TESS sector number (1, 2, 3, …).
        author: Data product author. ``"SPOC"`` is the Science Processing
            Operations Center pipeline (2-min cadence).
        limit: Maximum number of TIC IDs to return. ``None`` returns all.

    Returns:
        A list of TIC ID strings, e.g. ``["TIC 150428135", ...]``.
    """
    logger.info("Searching for targets in TESS sector %d (author=%s)", sector, author)
    search_result = lk.search_lightcurve(
        target=f"sector {sector}",
        mission="TESS",
        author=author,
    )

    # Extract unique TIC IDs from the search result table
    tic_ids: list[str] = []
    seen: set[str] = set()
    for row in search_result:
        tic_id = str(row.target_name)
        if tic_id not in seen:
            seen.add(tic_id)
            tic_ids.append(tic_id)

    if limit is not None:
        tic_ids = tic_ids[:limit]

    logger.info("Found %d unique targets in sector %d", len(tic_ids), sector)
    return tic_ids


def download_lightcurve(
    tic_id: str,
    sectors: list[int] | None = None,
    cache_dir: Path | None = None,
) -> LightCurve | None:
    """Download and stitch a TESS light curve for a single target.

    The function first checks the local cache. On a cache miss it queries
    MAST via lightkurve, downloads the data, stitches multi-sector
    observations, and saves the result to cache for future runs.

    Args:
        tic_id: Target identifier, e.g. ``"TIC 150428135"`` or ``"150428135"``.
        sectors: List of TESS sectors to include. ``None`` downloads all
            available sectors.
        cache_dir: Directory for cached light curves. Defaults to
            ``config.CACHE_DIR``.

    Returns:
        A stitched ``LightCurve`` object, or ``None`` if no data is found.
    """
    if cache_dir is None:
        cache_dir = config.CACHE_DIR

    # Normalize identifier — lightkurve expects "TIC <number>"
    if not tic_id.upper().startswith("TIC"):
        tic_id = f"TIC {tic_id}"

    # Try the local cache first
    cached = load_from_cache(tic_id, cache_dir)
    if cached is not None:
        logger.debug("Cache hit for %s", tic_id)
        return cached

    # Query MAST with retries
    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        try:
            logger.info(
                "Searching MAST for %s (attempt %d/%d)",
                tic_id,
                attempt,
                config.DOWNLOAD_MAX_RETRIES,
            )

            search_kwargs: dict = {
                "target": tic_id,
                "mission": "TESS",
                "author": "SPOC",
            }
            if sectors is not None:
                search_kwargs["sector"] = sectors

            search_result = lk.search_lightcurve(**search_kwargs)

            if len(search_result) == 0:
                logger.warning("No TESS data found for %s", tic_id)
                return None

            logger.info(
                "Downloading %d observations for %s",
                len(search_result),
                tic_id,
            )
            lc_collection = search_result.download_all()
            light_curve = lc_collection.stitch()

            # Persist to cache
            save_to_cache(light_curve, tic_id, cache_dir)
            return light_curve

        except Exception as exc:
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt,
                config.DOWNLOAD_MAX_RETRIES,
                tic_id,
                exc,
            )
            if attempt < config.DOWNLOAD_MAX_RETRIES:
                # Exponential back-off: 2s, 4s, 8s, …
                wait = 2**attempt
                logger.info("Retrying in %d seconds…", wait)
                time.sleep(wait)

    logger.error("All download attempts exhausted for %s", tic_id)
    return None


@timing
def download_batch(
    tic_ids: list[str],
    sectors: list[int] | None = None,
    max_workers: int = config.DEFAULT_DOWNLOAD_WORKERS,
) -> list[LightCurve]:
    """Download light curves for multiple targets concurrently.

    Uses threads because downloads are I/O-bound (network latency dominates).

    Args:
        tic_ids: List of TIC identifiers.
        sectors: TESS sectors to include (applied to every target).
        max_workers: Maximum number of concurrent download threads.

    Returns:
        A list of successfully downloaded ``LightCurve`` objects.
        Targets that failed to download are silently omitted.
    """

    def _download_one(tic_id: str) -> LightCurve | None:
        return download_lightcurve(tic_id, sectors=sectors)

    raw_results = run_parallel_threads(
        func=_download_one,
        items=tic_ids,
        max_workers=max_workers,
        description="Downloading light curves",
    )

    # Filter out None results (failed downloads)
    light_curves = [lc for lc in raw_results if lc is not None]
    logger.info(
        "Batch download complete: %d/%d targets succeeded",
        len(light_curves),
        len(tic_ids),
    )
    return light_curves
