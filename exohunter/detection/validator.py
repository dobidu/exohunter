"""Transit candidate validation.

After BLS detects a periodic signal, we apply a series of astrophysical
and statistical tests to decide whether the candidate is a genuine
planet, a false positive (e.g. eclipsing binary), or noise.

Validation criteria (following community standards):
    1. SNR ≥ 7.0
    2. Transit depth between 0.01% and 5%
    3. Duration consistent with period (Kepler's third law)
    4. At least 3 observed transits
    5. V-shape test (box-like = planet, V-shaped = binary)
    6. Harmonic check (period is not a harmonic of another candidate)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from exohunter import config
from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of candidate validation with details on each test.

    Attributes:
        is_valid: ``True`` if the candidate passes all critical tests.
        flags: List of human-readable warning/failure messages.
        tests: Dictionary mapping test names to pass/fail booleans.
    """

    is_valid: bool
    flags: list[str] = field(default_factory=list)
    tests: dict[str, bool] = field(default_factory=dict)


def validate_candidate(
    candidate: TransitCandidate,
    time: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    other_candidates: list[TransitCandidate] | None = None,
) -> ValidationResult:
    """Apply all validation tests to a transit candidate.

    Args:
        candidate: The ``TransitCandidate`` to validate.
        time: Time array of the light curve (needed for V-shape test).
        flux: Flux array of the light curve (needed for V-shape test).
        other_candidates: Other candidates from the same target, used
            for the harmonic check.

    Returns:
        A ``ValidationResult`` with overall pass/fail and per-test details.
    """
    flags: list[str] = []
    tests: dict[str, bool] = {}

    # --- Test 1: Signal-to-noise ratio ---
    snr_ok = candidate.snr >= config.MIN_SNR
    tests["snr"] = snr_ok
    if not snr_ok:
        flags.append(
            f"SNR too low: {candidate.snr:.1f} < {config.MIN_SNR}"
        )

    # --- Test 2: Transit depth bounds ---
    depth_ok = config.MIN_DEPTH <= candidate.depth <= config.MAX_DEPTH
    tests["depth"] = depth_ok
    if candidate.depth < config.MIN_DEPTH:
        flags.append(
            f"Transit too shallow: {candidate.depth:.6f} < {config.MIN_DEPTH} "
            f"(likely noise)"
        )
    elif candidate.depth > config.MAX_DEPTH:
        flags.append(
            f"Transit too deep: {candidate.depth:.4f} > {config.MAX_DEPTH} "
            f"(likely eclipsing binary)"
        )

    # --- Test 3: Duration consistency with period ---
    # For a circular orbit around a Sun-like star, the maximum transit
    # duration is approximately:  T_dur ≈ P/π × (R*/a)
    # For order-of-magnitude check, duration should be < ~25% of the period
    # and > ~0.1% of the period.
    duration_fraction = candidate.duration / candidate.period
    duration_ok = 0.001 < duration_fraction < 0.25
    tests["duration"] = duration_ok
    if not duration_ok:
        flags.append(
            f"Duration/period ratio suspicious: {duration_fraction:.4f} "
            f"(expected 0.001–0.25)"
        )

    # --- Test 4: Minimum number of transits ---
    transits_ok = candidate.n_transits >= config.MIN_TRANSITS
    tests["n_transits"] = transits_ok
    if not transits_ok:
        flags.append(
            f"Too few transits: {candidate.n_transits} < {config.MIN_TRANSITS}"
        )

    # --- Test 5: V-shape test ---
    # A genuine planetary transit has a flat bottom (box-like), while an
    # eclipsing binary produces a V-shaped dip.  We measure this by
    # comparing the depth at ingress/egress vs. the centre of transit.
    v_shape_ok = True
    if time is not None and flux is not None:
        v_metric = _compute_v_shape(
            time, flux, candidate.period, candidate.epoch, candidate.duration
        )
        v_shape_ok = v_metric <= config.MAX_V_SHAPE
        tests["v_shape"] = v_shape_ok
        if not v_shape_ok:
            flags.append(
                f"V-shape metric too high: {v_metric:.2f} > {config.MAX_V_SHAPE} "
                f"(possible eclipsing binary)"
            )
    else:
        tests["v_shape"] = True  # Skip if no data provided

    # --- Test 6: Harmonic check ---
    harmonic_ok = True
    if other_candidates:
        harmonic_ok = _check_harmonics(candidate, other_candidates)
        tests["harmonic"] = harmonic_ok
        if not harmonic_ok:
            flags.append(
                f"Period {candidate.period:.4f} d may be a harmonic of another candidate"
            )
    else:
        tests["harmonic"] = True

    # Overall result: must pass all critical tests
    # (V-shape and harmonic are warnings but don't invalidate)
    is_valid = all([snr_ok, depth_ok, duration_ok, transits_ok])

    result = ValidationResult(is_valid=is_valid, flags=flags, tests=tests)

    status = "VALID" if is_valid else "REJECTED"
    logger.info(
        "%s candidate %s (P=%.4f d): %s%s",
        status,
        candidate.tic_id,
        candidate.period,
        status,
        f" — {'; '.join(flags)}" if flags else "",
    )

    return result


def _compute_v_shape(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> float:
    """Compute the V-shape metric for a transit.

    The metric is defined as:
        V = 1 - (depth_at_center / depth_at_edge)

    where ``depth_at_center`` is measured at the transit midpoint and
    ``depth_at_edge`` is measured at ingress/egress.  A perfect box
    transit gives V ≈ 0; a perfect V gives V ≈ 1.

    Args:
        time: Array of timestamps.
        flux: Array of flux values.
        period: Orbital period in days.
        epoch: Mid-transit time.
        duration: Transit duration in days.

    Returns:
        V-shape metric between 0.0 (box) and 1.0 (V).
    """
    # Phase-fold and isolate transit points
    phase = ((time - epoch) % period) / period
    # Shift so that transit is centred at phase = 0.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    half_dur_phase = (duration / period) / 2.0

    # Points in the central third of the transit
    center_mask = np.abs(phase) < half_dur_phase / 3.0
    # Points in the outer thirds (ingress/egress)
    edge_mask = (np.abs(phase) > half_dur_phase / 3.0) & (
        np.abs(phase) < half_dur_phase
    )

    if np.sum(center_mask) < 3 or np.sum(edge_mask) < 3:
        # Not enough data — assume box-like (conservative)
        return 0.0

    center_flux = float(np.median(flux[center_mask]))
    edge_flux = float(np.median(flux[edge_mask]))
    baseline = 1.0  # Normalised flux

    depth_center = baseline - center_flux
    depth_edge = baseline - edge_flux

    if depth_center <= 0:
        return 0.0

    v_metric = 1.0 - (depth_edge / depth_center)
    return max(0.0, min(1.0, v_metric))


def _check_harmonics(
    candidate: TransitCandidate,
    others: list[TransitCandidate],
    tolerance: float = 0.01,
) -> bool:
    """Check if a candidate's period is a harmonic of another candidate.

    Tests ratios 1:2, 2:1, 1:3, 3:1.

    Args:
        candidate: The candidate to test.
        others: Other candidates from the same target.
        tolerance: Relative tolerance for period ratio matching.

    Returns:
        ``True`` if the period is NOT a harmonic (test passed).
    """
    harmonic_ratios = [0.5, 2.0, 1.0 / 3.0, 3.0]

    for other in others:
        if other is candidate:
            continue
        ratio = candidate.period / other.period
        for h in harmonic_ratios:
            if abs(ratio - h) / h < tolerance:
                logger.debug(
                    "Period %.4f d is a %.0f:1 harmonic of %.4f d",
                    candidate.period,
                    1 / h if h < 1 else h,
                    other.period,
                )
                return False

    return True
