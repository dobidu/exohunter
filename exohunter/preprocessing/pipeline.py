"""Preprocessing pipeline orchestrator.

Combines cleaning, normalization, detrending, and quality estimation
into a single ``preprocess_single`` function.  The ``preprocess_batch``
function runs this pipeline on multiple light curves in parallel using
a ``ProcessPoolExecutor`` (CPU-bound work benefits from multiple cores).

Dataflow::

    raw LightCurve
      → remove NaN
      → remove outliers (5σ)
      → normalize (median → 1.0)
      → flatten (remove stellar variability)
      → estimate CDPP noise metric
      → ProcessedLightCurve
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from lightkurve import LightCurve

from exohunter import config
from exohunter.preprocessing.clean import remove_nans, remove_outliers
from exohunter.preprocessing.detrend import flatten_lightcurve
from exohunter.preprocessing.normalize import normalize_lightcurve
from exohunter.utils.logging import get_logger
from exohunter.utils.parallel import run_parallel_processes
from exohunter.utils.timing import timing

logger = get_logger(__name__)


@dataclass
class ProcessedLightCurve:
    """Container for a preprocessed light curve and its metadata.

    Attributes:
        time: Array of timestamps (BTJD — Barycentric TESS Julian Date).
        flux: Array of normalized, detrended flux values.
        flux_err: Array of flux uncertainties (propagated through pipeline).
        cdpp: Combined Differential Photometric Precision in ppm — a noise
            metric that estimates how well we can detect transits.
        tic_id: Target identifier (e.g. ``"TIC 150428135"``).
        sectors: List of TESS sectors included in this light curve.
        metadata: Additional metadata from the FITS headers.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    cdpp: float
    tic_id: str
    sectors: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_lightcurve(self) -> LightCurve:
        """Convert back to a lightkurve ``LightCurve`` object.

        Useful for running BLS or other lightkurve methods on
        preprocessed data.
        """
        return LightCurve(
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
        )


def preprocess_single(
    light_curve: LightCurve,
    tic_id: str = "unknown",
) -> ProcessedLightCurve:
    """Apply the full preprocessing pipeline to a single light curve.

    Steps:
        1. Remove NaN cadences
        2. Remove outliers (5σ clipping)
        3. Normalize (median flux → 1.0)
        4. Flatten (Savitzky-Golay detrending)
        5. Estimate CDPP noise metric

    Args:
        light_curve: Raw ``LightCurve`` from lightkurve download.
        tic_id: Target identifier for labelling.

    Returns:
        A ``ProcessedLightCurve`` dataclass with arrays and metadata.
    """
    logger.info("Preprocessing %s (%d cadences)", tic_id, len(light_curve))

    # Step 1 — Remove NaN values
    lc = remove_nans(light_curve)

    # Step 2 — Remove flux outliers (cosmic rays, glitches)
    lc = remove_outliers(lc, sigma=config.OUTLIER_SIGMA)

    # Step 3 — Normalize to median flux = 1.0
    lc = normalize_lightcurve(lc)

    # Step 4 — Remove long-period stellar variability
    lc = flatten_lightcurve(lc, window_length=config.FLATTEN_WINDOW_LENGTH)

    # Step 5 — Estimate noise (CDPP = Combined Differential Photometric Precision)
    # CDPP tells us the expected transit detection sensitivity in ppm.
    try:
        cdpp_value = float(lc.estimate_cdpp(transit_duration=config.CDPP_TRANSIT_DURATION_HOURS))
    except Exception:
        logger.warning("Could not estimate CDPP for %s — using NaN", tic_id)
        cdpp_value = float("nan")

    # Extract sector information from metadata if available
    sectors: list[int] = []
    if hasattr(light_curve, "meta") and "SECTOR" in light_curve.meta:
        sectors = [int(light_curve.meta["SECTOR"])]

    logger.info(
        "Preprocessing complete for %s: %d cadences, CDPP=%.1f ppm",
        tic_id,
        len(lc),
        cdpp_value,
    )

    return ProcessedLightCurve(
        time=np.array(lc.time.value, dtype=np.float64),
        flux=np.array(lc.flux.value, dtype=np.float64),
        flux_err=np.array(lc.flux_err.value, dtype=np.float64) if lc.flux_err is not None else np.zeros_like(lc.flux.value),
        cdpp=cdpp_value,
        tic_id=tic_id,
        sectors=sectors,
        metadata=dict(lc.meta) if hasattr(lc, "meta") else {},
    )


def _preprocess_wrapper(args: tuple[LightCurve, str]) -> ProcessedLightCurve:
    """Pickle-friendly wrapper for ``preprocess_single``.

    ``ProcessPoolExecutor`` requires top-level functions (not closures)
    and can only pass a single argument to the target function, so we
    pack the arguments into a tuple.
    """
    light_curve, tic_id = args
    return preprocess_single(light_curve, tic_id=tic_id)


@timing
def preprocess_batch(
    lightcurves: list[tuple[LightCurve, str]],
    max_workers: int | None = None,
) -> list[ProcessedLightCurve]:
    """Preprocess multiple light curves in parallel using processes.

    CPU-bound work (flattening, outlier removal) benefits from true
    parallelism across multiple cores.

    Args:
        lightcurves: List of ``(LightCurve, tic_id)`` tuples.
        max_workers: Number of worker processes.
            Defaults to ``os.cpu_count() - 1``.

    Returns:
        A list of ``ProcessedLightCurve`` objects (failed items omitted).
    """
    if max_workers is None:
        cpu_count = os.cpu_count() or 2
        max_workers = max(1, cpu_count - 1)

    results = run_parallel_processes(
        func=_preprocess_wrapper,
        items=lightcurves,
        max_workers=max_workers,
        description="Preprocessing light curves",
    )

    # Filter out None entries from failed items
    return [r for r in results if r is not None]
