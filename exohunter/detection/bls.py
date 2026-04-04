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
#
# This uses a binned cumulative-sum approach (similar to astropy's C
# implementation) instead of brute-force sliding.  For each trial
# period:
#   1. Phase-fold and sort the data.
#   2. Bin the phase into N_BINS uniform bins, accumulating flux sums
#      and counts in each bin.
#   3. Build prefix sums over the bins.
#   4. For each trial duration, slide a window of the appropriate
#      number of bins and compute the in-transit / out-of-transit
#      means in O(1) per position using the prefix sums.
#
# Complexity: O(n_periods × (n_points + n_bins × n_durations))
# vs. the brute-force O(n_periods × n_bins × n_durations × n_points).
# ---------------------------------------------------------------------------

try:
    import numba

    # Number of phase bins — controls the epoch resolution.
    # 300 bins ≈ 0.3% of the period, fine enough for TESS 2-min cadence.
    _N_PHASE_BINS: int = 300

    @numba.njit(parallel=True)
    def _bls_core(
        time: np.ndarray,
        flux: np.ndarray,
        periods: np.ndarray,
        durations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core BLS computation accelerated with Numba.

        Uses a binned prefix-sum algorithm: for each trial period, the
        phase-folded data is binned, then a sliding window over the
        bin sums evaluates every trial epoch in O(1) per position.

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
        n_bins = _N_PHASE_BINS

        powers = np.zeros(n_periods, dtype=np.float64)
        depths = np.zeros(n_periods, dtype=np.float64)
        best_durations = np.zeros(n_periods, dtype=np.float64)
        epochs = np.zeros(n_periods, dtype=np.float64)

        total_flux = 0.0
        for k in range(n_points):
            total_flux += flux[k]

        for i in numba.prange(n_periods):
            period = periods[i]
            best_power = 0.0
            best_depth = 0.0
            best_dur = durations[0]
            best_epoch = 0.0

            # --- Step 1: Bin the phase-folded data ---
            # Each bin accumulates the sum of flux values and a count
            # of how many points fell into it.
            bin_flux = np.zeros(n_bins, dtype=np.float64)
            bin_count = np.zeros(n_bins, dtype=np.int64)

            for k in range(n_points):
                phase = (time[k] % period) / period
                b = int(phase * n_bins)
                if b >= n_bins:
                    b = n_bins - 1
                bin_flux[b] += flux[k]
                bin_count[b] += 1

            # --- Step 2: For each trial duration, slide over bins ---
            for j in range(len(durations)):
                dur = durations[j]
                frac_dur = dur / period

                if frac_dur >= 0.5:
                    continue

                # How many bins does this duration span?
                w = max(1, int(frac_dur * n_bins + 0.5))

                # Initialize the first window [0, w)
                s_in = 0.0
                c_in = 0
                for b in range(w):
                    s_in += bin_flux[b]
                    c_in += bin_count[b]

                # Slide the window across all n_bins starting positions
                for b in range(n_bins):
                    c_out = n_points - c_in
                    if c_in >= 3 and c_out >= 3:
                        mean_in = s_in / c_in
                        mean_out = (total_flux - s_in) / c_out
                        delta = mean_out - mean_in

                        if delta > 0.0:
                            power = (c_in * c_out * delta * delta) / n_points
                            if power > best_power:
                                best_power = power
                                best_depth = delta
                                best_dur = dur
                                # Epoch = center of the window
                                center_bin = (b + w / 2.0) % n_bins
                                best_epoch = (center_bin / n_bins) * period

                    # Slide: remove the leftmost bin, add the next one
                    s_in -= bin_flux[b % n_bins]
                    c_in -= bin_count[b % n_bins]
                    add_bin = (b + w) % n_bins
                    s_in += bin_flux[add_bin]
                    c_in += bin_count[add_bin]

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


# ---------------------------------------------------------------------------
# GPU-accelerated BLS (Numba CUDA)
# ---------------------------------------------------------------------------
#
# The same prefix-sum binned algorithm as _bls_core, but each trial
# period is processed by a separate CUDA thread instead of a CPU core.
# With 10,000 periods, this maps naturally to the GPU's thousands of
# threads.
#
# Requires: NVIDIA GPU with CUDA support (numba.cuda).
# Falls back to CPU Numba transparently if no GPU is detected.
# ---------------------------------------------------------------------------

try:
    from numba import cuda as _cuda

    _CUDA_AVAILABLE = _cuda.is_available()
except (ImportError, Exception):
    _CUDA_AVAILABLE = False

if _NUMBA_AVAILABLE and _CUDA_AVAILABLE:
    from numba import cuda as _cuda

    # GPU uses 128 bins — fewer than the CPU's 300 to keep memory
    # usage per thread low.  128 bins ≈ 0.8% of the period, still
    # fine for TESS 2-min cadence data.
    _N_GPU_BINS: int = 128

    @_cuda.jit
    def _bls_core_gpu(
        time, flux, periods, durations,
        n_points, n_durations, n_bins, total_flux,
        work_flux, work_count,
        powers_out, depths_out, durations_out, epochs_out,
    ):
        """CUDA kernel: one thread per trial period.

        Each thread uses a pre-allocated row in the global work arrays
        (work_flux[i,:] and work_count[i,:]) for its bin data, avoiding
        local memory allocation that can crash on smaller GPUs.
        """
        i = _cuda.grid(1)
        if i >= periods.shape[0]:
            return

        period = periods[i]
        best_power = 0.0
        best_depth = 0.0
        best_dur = durations[0]
        best_epoch = 0.0

        # Zero the bin arrays (each thread owns row i)
        for b in range(n_bins):
            work_flux[i, b] = 0.0
            work_count[i, b] = 0

        # Step 1: Bin the phase-folded data
        for k in range(n_points):
            phase = (time[k] % period) / period
            b = int(phase * n_bins)
            if b >= n_bins:
                b = n_bins - 1
            work_flux[i, b] += flux[k]
            work_count[i, b] += 1

        # Step 2: Slide window for each duration
        for j in range(n_durations):
            dur = durations[j]
            frac_dur = dur / period

            if frac_dur >= 0.5:
                continue

            w = max(1, int(frac_dur * n_bins + 0.5))

            s_in = 0.0
            c_in = 0
            for b in range(w):
                s_in += work_flux[i, b]
                c_in += work_count[i, b]

            for b in range(n_bins):
                c_out = n_points - c_in
                if c_in >= 3 and c_out >= 3:
                    mean_in = s_in / c_in
                    mean_out = (total_flux - s_in) / c_out
                    delta = mean_out - mean_in

                    if delta > 0.0:
                        power = (c_in * c_out * delta * delta) / n_points
                        if power > best_power:
                            best_power = power
                            best_depth = delta
                            best_dur = dur
                            center_bin = (b + w / 2.0) % n_bins
                            best_epoch = (center_bin / n_bins) * period

                s_in -= work_flux[i, b % n_bins]
                c_in -= work_count[i, b % n_bins]
                add_bin = (b + w) % n_bins
                s_in += work_flux[i, add_bin]
                c_in += work_count[i, add_bin]

        powers_out[i] = best_power
        depths_out[i] = best_depth
        durations_out[i] = best_dur
        epochs_out[i] = best_epoch


@timing
def run_bls_gpu(
    time: np.ndarray,
    flux: np.ndarray,
    tic_id: str = "unknown",
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
    num_periods: int = config.BLS_NUM_PERIODS,
    durations_hours: list[float] | None = None,
) -> TransitCandidate | None:
    """Run BLS transit search on GPU via Numba CUDA.

    If no CUDA GPU is detected, falls back to the CPU Numba
    implementation transparently.

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
    if not _CUDA_AVAILABLE:
        logger.info("No CUDA GPU detected — falling back to CPU Numba BLS")
        return run_bls_numba(
            time, flux, tic_id=tic_id,
            min_period=min_period, max_period=max_period,
            num_periods=num_periods, durations_hours=durations_hours,
        )

    if durations_hours is None:
        durations_hours = config.BLS_DURATIONS_HOURS

    logger.info("Running BLS (GPU/CUDA) on %s", tic_id)

    # Wrap the entire GPU execution in a safety net.
    # Numba CUDA can crash (segfault) on some environments (e.g. WSL2
    # with certain driver versions) during kernel compilation.  If
    # anything goes wrong, fall back to CPU Numba transparently.
    try:
        return _run_bls_gpu_inner(
            time, flux, tic_id, min_period, max_period,
            num_periods, durations_hours,
        )
    except Exception as exc:
        logger.warning(
            "GPU BLS failed (%s) — falling back to CPU Numba", exc
        )
        return run_bls_numba(
            time, flux, tic_id=tic_id,
            min_period=min_period, max_period=max_period,
            num_periods=num_periods, durations_hours=durations_hours,
        )


def _run_bls_gpu_inner(
    time: np.ndarray,
    flux: np.ndarray,
    tic_id: str,
    min_period: float,
    max_period: float,
    num_periods: int,
    durations_hours: list[float],
) -> TransitCandidate | None:
    """Inner GPU BLS implementation (called by run_bls_gpu with safety wrapper)."""
    periods = np.linspace(min_period, max_period, num_periods)
    durations_days = np.array([d / 24.0 for d in durations_hours])

    n_points = len(time)
    n_durations = len(durations_days)
    n_bins = _N_GPU_BINS
    total_flux = float(np.sum(flux))

    # Transfer data to GPU
    d_time = _cuda.to_device(time.astype(np.float64))
    d_flux = _cuda.to_device(flux.astype(np.float64))
    d_periods = _cuda.to_device(periods.astype(np.float64))
    d_durations = _cuda.to_device(durations_days.astype(np.float64))

    # Output arrays on GPU
    d_powers = _cuda.device_array(num_periods, dtype=np.float64)
    d_depths = _cuda.device_array(num_periods, dtype=np.float64)
    d_best_durs = _cuda.device_array(num_periods, dtype=np.float64)
    d_epochs = _cuda.device_array(num_periods, dtype=np.float64)

    # Work arrays for per-thread bin data (global memory, not local)
    d_work_flux = _cuda.device_array((num_periods, n_bins), dtype=np.float64)
    d_work_count = _cuda.device_array((num_periods, n_bins), dtype=np.int64)

    # Launch kernel: 128 threads per block (conservative for memory)
    threads_per_block = 128
    blocks = (num_periods + threads_per_block - 1) // threads_per_block

    _bls_core_gpu[blocks, threads_per_block](
        d_time, d_flux, d_periods, d_durations,
        n_points, n_durations, n_bins, total_flux,
        d_work_flux, d_work_count,
        d_powers, d_depths, d_best_durs, d_epochs,
    )

    # Copy results back to CPU
    powers = d_powers.copy_to_host()
    depths_arr = d_depths.copy_to_host()
    best_durations = d_best_durs.copy_to_host()
    epochs_arr = d_epochs.copy_to_host()

    # Find best period
    best_idx = int(np.argmax(powers))
    best_period = float(periods[best_idx])
    best_depth = float(depths_arr[best_idx])
    best_duration = float(best_durations[best_idx])
    best_epoch = float(epochs_arr[best_idx])
    best_power = float(powers[best_idx])

    if best_depth <= 0:
        logger.warning("No transit signal found for %s (GPU)", tic_id)
        return None

    # Estimate SNR
    phase = (time % best_period) / best_period
    fractional_dur = best_duration / best_period
    phase_center = best_epoch / best_period
    diff = np.abs(phase - phase_center)
    diff = np.minimum(diff, 1.0 - diff)
    out_mask = diff >= fractional_dur / 2.0
    out_flux = flux[out_mask]
    scatter = float(np.std(out_flux)) if len(out_flux) > 0 else 1.0
    n_in = int(np.sum(~out_mask))
    snr = best_depth / scatter * np.sqrt(n_in) if scatter > 0 else 0.0

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
        "BLS (GPU) result for %s: P=%.4f d, depth=%.4f%%, SNR=%.1f",
        tic_id, best_period, best_depth * 100, snr,
    )
    return candidate


# ---------------------------------------------------------------------------
# Iterative multi-planet BLS search
# ---------------------------------------------------------------------------

def run_iterative_bls(
    light_curve: LightCurve,
    tic_id: str = "unknown",
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
    num_periods: int = config.BLS_NUM_PERIODS,
    frequency_factor: int = config.BLS_FREQUENCY_FACTOR,
    max_planets: int = 5,
    min_snr: float = 5.0,
) -> list[TransitCandidate]:
    """Search for multiple transiting planets by iterative subtraction.

    Algorithm:
        1. Run BLS on the light curve to find the strongest signal.
        2. If the signal has SNR >= ``min_snr``, record the candidate.
        3. Subtract the best-fit trapezoidal transit model from the flux.
        4. Re-run BLS on the residual light curve.
        5. Repeat until no signal exceeds ``min_snr`` or ``max_planets``
           candidates have been found.

    The subtraction uses the trapezoidal model from
    ``exohunter.detection.model.transit_model``, which provides a
    good-enough approximation for removing the transit signal without
    overfitting.

    Args:
        light_curve: Preprocessed ``LightCurve`` (normalized, detrended).
        tic_id: Target identifier.
        min_period: BLS minimum trial period (days).
        max_period: BLS maximum trial period (days).
        num_periods: Number of trial periods in the BLS grid.
        frequency_factor: BLS frequency oversampling factor.
        max_planets: Maximum number of planets to search for.
        min_snr: Minimum SNR to accept a candidate. Set lower than the
            validation threshold (7.0) to catch weaker signals that
            may still be real planets.

    Returns:
        A list of ``TransitCandidate`` objects, ordered by detection
        (strongest signal first). May be empty if no signal exceeds
        ``min_snr``.
    """
    from exohunter.detection.model import transit_model

    candidates: list[TransitCandidate] = []

    # Work on copies of the time and flux arrays so we don't modify
    # the original LightCurve.
    time = np.array(light_curve.time.value, dtype=np.float64)
    flux = np.array(light_curve.flux.value, dtype=np.float64)

    logger.info(
        "Starting iterative BLS on %s (max %d planets, min SNR %.1f)",
        tic_id, max_planets, min_snr,
    )

    for iteration in range(max_planets):
        # Build a LightCurve from the current (possibly residual) flux
        residual_lc = LightCurve(time=time, flux=flux)

        candidate = run_bls_lightkurve(
            residual_lc,
            tic_id=tic_id,
            min_period=min_period,
            max_period=max_period,
            num_periods=num_periods,
            frequency_factor=frequency_factor,
        )

        if candidate is None:
            logger.info("Iteration %d: BLS returned no candidate — stopping", iteration + 1)
            break

        if candidate.snr < min_snr:
            logger.info(
                "Iteration %d: SNR=%.1f < %.1f — stopping",
                iteration + 1, candidate.snr, min_snr,
            )
            break

        # Check this isn't a duplicate of an already-found period
        # (within 1% or an exact harmonic)
        is_duplicate = False
        for prev in candidates:
            ratio = candidate.period / prev.period
            for harmonic in [1.0, 0.5, 2.0, 1.0 / 3.0, 3.0]:
                if abs(ratio - harmonic) / harmonic < 0.01:
                    is_duplicate = True
                    break
            if is_duplicate:
                break

        if is_duplicate:
            logger.info(
                "Iteration %d: P=%.4f d is a duplicate/harmonic of a "
                "previous detection — stopping",
                iteration + 1, candidate.period,
            )
            break

        # Label the candidate with its planet letter (b, c, d, ...)
        planet_letter = chr(ord("b") + iteration)
        candidate.name = f"{tic_id} {planet_letter}"

        candidates.append(candidate)
        logger.info(
            "Iteration %d: found planet %s — P=%.4f d, depth=%.4f%%, SNR=%.1f",
            iteration + 1, planet_letter,
            candidate.period, candidate.depth * 100, candidate.snr,
        )

        # Subtract the transit model from the flux
        model_flux = transit_model(
            time=time,
            period=candidate.period,
            epoch=candidate.epoch,
            duration=candidate.duration,
            depth=candidate.depth,
        )
        # The model produces values like 1.0 (out of transit) and
        # 1.0-depth (in transit).  To subtract: flux_residual = flux - (model - 1.0)
        # This removes the transit dip and leaves the baseline at ~1.0.
        flux = flux - (model_flux - 1.0)

    logger.info(
        "Iterative BLS on %s: found %d planet(s)",
        tic_id, len(candidates),
    )
    return candidates
