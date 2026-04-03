"""Box Least Squares (BLS) transit detection.

Two implementations are provided:

1. **lightkurve wrapper** (``run_bls_lightkurve``): Uses the well-tested
   astropy BLS implementation exposed through lightkurve.  This is the
   default for production use.

2. **Numba implementation** (``run_bls_numba``): A from-scratch BLS
   written in pure Python and accelerated with ``@numba.njit``.  This
   exists for pedagogical purposes (students can study the algorithm)
   and as a performance experiment.

Background — What is BLS?
    The Box Least Squares algorithm (Kovács, Zucker & Mazeh 2002)
    searches for periodic box-shaped dips in a time series.  For each
    trial period *P* and duration *d*, it finds the epoch *t₀* and
    depth *δ* that minimise the residuals, producing a "power"
    spectrum over the period grid.  The period with the highest power
    is the best transit candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightkurve import LightCurve

from exohunter import config
from exohunter.utils.logging import get_logger
from exohunter.utils.timing import timing

logger = get_logger(__name__)


@dataclass
class TransitCandidate:
    """A detected transit candidate and its parameters.

    Attributes:
        tic_id: Target identifier.
        period: Orbital period in days.
        epoch: Mid-transit time (t₀) of the first transit, in BTJD.
        duration: Transit duration in days.
        depth: Fractional transit depth (e.g. 0.01 = 1%).
        snr: Signal-to-noise ratio of the detection.
        bls_power: Peak BLS power spectrum value.
        n_transits: Number of individual transit events observed.
        name: Optional human-readable name (e.g. ``"TOI-700 d"``).
    """

    tic_id: str
    period: float
    epoch: float
    duration: float
    depth: float
    snr: float
    bls_power: float
    n_transits: int = 0
    name: str = ""


@timing
def run_bls_lightkurve(
    light_curve: LightCurve,
    tic_id: str = "unknown",
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
    num_periods: int = config.BLS_NUM_PERIODS,
    frequency_factor: int = config.BLS_FREQUENCY_FACTOR,
) -> TransitCandidate | None:
    """Run BLS transit search using lightkurve/astropy.

    This is the recommended method for production use.  It wraps
    astropy's optimised BLS implementation with lightkurve's
    convenient interface.

    Args:
        light_curve: Preprocessed ``LightCurve`` (normalized, detrended).
        tic_id: Target identifier for labelling.
        min_period: Minimum trial period in days.
        max_period: Maximum trial period in days.
        num_periods: Number of periods to search.
        frequency_factor: Oversampling factor for the frequency grid.

    Returns:
        A ``TransitCandidate`` if a signal is found, or ``None`` on failure.
    """
    logger.info("Running BLS (lightkurve) on %s", tic_id)

    try:
        period_grid = np.linspace(min_period, max_period, num_periods)

        periodogram = light_curve.to_periodogram(
            method="bls",
            period=period_grid,
            frequency_factor=frequency_factor,
        )

        best_period = float(periodogram.period_at_max_power.value)
        best_t0 = float(periodogram.transit_time_at_max_power.value)
        best_duration = float(periodogram.duration_at_max_power.value)
        best_depth = float(periodogram.depth_at_max_power.value)
        best_power = float(periodogram.max_power.value)

        # Compute additional statistics
        stats = periodogram.compute_stats(
            period=best_period,
            duration=best_duration,
            transit_time=best_t0,
        )

        # SNR: depth divided by the uncertainty on the depth
        depth_err = stats.get("depth_err", [0, 0])
        if isinstance(depth_err, (list, np.ndarray)) and len(depth_err) > 0:
            avg_depth_err = float(np.mean(np.abs(depth_err)))
        else:
            avg_depth_err = float(depth_err) if depth_err else 0.0

        snr = best_depth / avg_depth_err if avg_depth_err > 0 else 0.0

        # Count transits: how many complete periods fit in the observation span
        time_span = float(light_curve.time.value[-1] - light_curve.time.value[0])
        n_transits = max(1, int(time_span / best_period))

        candidate = TransitCandidate(
            tic_id=tic_id,
            period=best_period,
            epoch=best_t0,
            duration=best_duration,
            depth=best_depth,
            snr=snr,
            bls_power=best_power,
            n_transits=n_transits,
        )

        logger.info(
            "BLS result for %s: P=%.4f d, depth=%.4f%%, SNR=%.1f",
            tic_id,
            best_period,
            best_depth * 100,
            snr,
        )
        return candidate

    except Exception:
        logger.exception("BLS (lightkurve) failed for %s", tic_id)
        return None


# ---------------------------------------------------------------------------
# Numba implementation — pedagogical from-scratch BLS
# ---------------------------------------------------------------------------

try:
    import numba

    @numba.njit(parallel=True)
    def _bls_core(
        time: np.ndarray,
        flux: np.ndarray,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core BLS computation accelerated with Numba.

        For each trial period and duration, this function phase-folds the
        light curve and finds the box depth that minimises the squared
        residuals.

        The algorithm:
            1. For each period P in the grid:
               a. Phase-fold the time array: φ = (t mod P) / P
               b. For each duration d:
                  - Slide a box of width d/P across the phase
                  - Compute the mean in-transit and out-of-transit flux
                  - The "power" is proportional to n_in × n_out × δ²
               c. Keep the (duration, phase) that maximises the power
            2. Return the best power, depth, duration, and epoch for
               each trial period.

        Args:
            time: Array of timestamps.
            flux: Array of flux values (normalised around 1.0).
            periods: 1-D array of trial periods.
            durations: 1-D array of trial durations (same units as periods).

        Returns:
            Tuple of (powers, depths, best_durations, epochs) — each an
            array of length ``len(periods)``.
        """
        n_periods = len(periods)
        n_points = len(time)

        powers = np.zeros(n_periods, dtype=np.float64)
        depths = np.zeros(n_periods, dtype=np.float64)
        best_durations = np.zeros(n_periods, dtype=np.float64)
        epochs = np.zeros(n_periods, dtype=np.float64)

        for i in numba.prange(n_periods):
            period = periods[i]
            best_power = 0.0
            best_depth = 0.0
            best_dur = durations[0]
            best_epoch = 0.0

            # Phase-fold the time series
            phase = np.empty(n_points, dtype=np.float64)
            for k in range(n_points):
                phase[k] = (time[k] % period) / period

            for j in range(len(durations)):
                duration = durations[j]
                fractional_duration = duration / period

                if fractional_duration >= 0.5:
                    # Duration too long for this period — skip
                    continue

                # Number of phase bins to slide
                n_bins = 200
                for b in range(n_bins):
                    phase_center = b / n_bins

                    # Count in-transit and out-of-transit sums
                    sum_in = 0.0
                    sum_out = 0.0
                    n_in = 0
                    n_out = 0

                    for k in range(n_points):
                        # Distance in phase (wrapping around)
                        diff = abs(phase[k] - phase_center)
                        if diff > 0.5:
                            diff = 1.0 - diff

                        if diff < fractional_duration / 2.0:
                            sum_in += flux[k]
                            n_in += 1
                        else:
                            sum_out += flux[k]
                            n_out += 1

                    if n_in < 3 or n_out < 3:
                        continue

                    mean_in = sum_in / n_in
                    mean_out = sum_out / n_out
                    delta = mean_out - mean_in  # transit depth

                    if delta <= 0:
                        continue

                    # BLS power ~ n_in × n_out × δ² / (n_in + n_out)
                    power = (n_in * n_out * delta * delta) / (n_in + n_out)

                    if power > best_power:
                        best_power = power
                        best_depth = delta
                        best_dur = duration
                        best_epoch = phase_center * period

            powers[i] = best_power
            depths[i] = best_depth
            best_durations[i] = best_dur
            epochs[i] = best_epoch

        return powers, depths, best_durations, epochs

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False
    logger.info("Numba not available — run_bls_numba will fall back to lightkurve")


@timing
def run_bls_numba(
    time: np.ndarray,
    flux: np.ndarray,
    tic_id: str = "unknown",
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
    num_periods: int = config.BLS_NUM_PERIODS,
    durations_hours: list[float] | None = None,
) -> TransitCandidate | None:
    """Run BLS transit search using the Numba-accelerated implementation.

    This is the pedagogical implementation — students can read and modify
    the ``_bls_core`` function to understand the BLS algorithm.

    Args:
        time: Array of timestamps (BTJD).
        flux: Array of normalized flux values.
        tic_id: Target identifier.
        min_period: Minimum trial period in days.
        max_period: Maximum trial period in days.
        num_periods: Number of trial periods.
        durations_hours: List of trial durations in hours.

    Returns:
        A ``TransitCandidate`` if a signal is found, or ``None``.
    """
    if not _NUMBA_AVAILABLE:
        logger.warning("Numba not available — falling back to lightkurve BLS")
        lc = LightCurve(time=time, flux=flux)
        return run_bls_lightkurve(lc, tic_id=tic_id)

    if durations_hours is None:
        durations_hours = config.BLS_DURATIONS_HOURS

    logger.info("Running BLS (Numba) on %s", tic_id)

    periods = np.linspace(min_period, max_period, num_periods)
    durations_days = np.array([d / 24.0 for d in durations_hours])

    powers, depths_arr, best_durations, epochs = _bls_core(
        time, flux, periods, durations_days
    )

    # Find the best period
    best_idx = int(np.argmax(powers))
    best_period = float(periods[best_idx])
    best_depth = float(depths_arr[best_idx])
    best_duration = float(best_durations[best_idx])
    best_epoch = float(epochs[best_idx])
    best_power = float(powers[best_idx])

    if best_depth <= 0:
        logger.warning("No transit signal found for %s", tic_id)
        return None

    # Estimate SNR: depth / scatter of out-of-transit flux
    # Phase-fold and compute scatter
    phase = (time % best_period) / best_period
    fractional_dur = best_duration / best_period
    phase_center = best_epoch / best_period
    diff = np.abs(phase - phase_center)
    diff = np.minimum(diff, 1.0 - diff)
    out_of_transit_mask = diff >= fractional_dur / 2.0
    out_of_transit_flux = flux[out_of_transit_mask]
    scatter = float(np.std(out_of_transit_flux)) if len(out_of_transit_flux) > 0 else 1.0

    # SNR accounts for the number of in-transit points
    n_in_transit = int(np.sum(~out_of_transit_mask))
    snr = best_depth / scatter * np.sqrt(n_in_transit) if scatter > 0 else 0.0

    # Count transits
    time_span = float(time[-1] - time[0])
    n_transits = max(1, int(time_span / best_period))

    candidate = TransitCandidate(
        tic_id=tic_id,
        period=best_period,
        epoch=best_epoch,
        duration=best_duration,
        depth=best_depth,
        snr=float(snr),
        bls_power=best_power,
        n_transits=n_transits,
    )

    logger.info(
        "BLS (Numba) result for %s: P=%.4f d, depth=%.4f%%, SNR=%.1f",
        tic_id,
        best_period,
        best_depth * 100,
        snr,
    )
    return candidate
