"""Simplified transit model for fitting and visualization.

Uses a trapezoidal approximation of a planetary transit:

    1.0  ─────┐         ┌─────  (out of transit)
              │╲       ╱│
              │ ╲     ╱ │       (ingress / egress)
    1-depth ──│──╲───╱──│──     (in-transit flat bottom)
              │   │   │  │
           t1  t2  t3  t4

The four contact times define:
    - t1→t2: ingress (star enters shadow)
    - t2→t3: flat bottom (full transit)
    - t3→t4: egress (star exits shadow)
    - Total duration = t4 - t1
    - Flat duration = t3 - t2

For a real planet, the ingress/egress time depends on the planet-to-star
radius ratio and the impact parameter, but for a first approximation
we model it as a fixed fraction of the total duration.
"""

import numpy as np

from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# Fraction of transit duration spent in ingress/egress (each side)
# For a typical hot Jupiter this is ~10-15% of total duration
DEFAULT_INGRESS_FRACTION: float = 0.1


def transit_model(
    time: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    ingress_fraction: float = DEFAULT_INGRESS_FRACTION,
) -> np.ndarray:
    """Generate a trapezoidal transit model.

    Args:
        time: Array of timestamps at which to evaluate the model.
        period: Orbital period in days.
        epoch: Mid-transit time (t₀) in the same time system as ``time``.
        duration: Total transit duration in days (t1 to t4).
        depth: Fractional transit depth (e.g. 0.01 for a 1% dip).
        ingress_fraction: Fraction of the total duration occupied by
            ingress (and symmetrically by egress).

    Returns:
        Array of model flux values (same length as ``time``).
    """
    # Phase-fold and centre the transit at phase = 0
    phase = ((time - epoch + period / 2) % period - period / 2)

    half_duration = duration / 2.0
    ingress_time = duration * ingress_fraction

    model_flux = np.ones_like(time)

    for i in range(len(time)):
        abs_phase = abs(phase[i])

        if abs_phase > half_duration:
            # Out of transit
            model_flux[i] = 1.0

        elif abs_phase < half_duration - ingress_time:
            # Full transit (flat bottom)
            model_flux[i] = 1.0 - depth

        else:
            # Ingress or egress — linear interpolation
            fraction = (half_duration - abs_phase) / ingress_time
            model_flux[i] = 1.0 - depth * fraction

    return model_flux


def transit_model_from_candidate(
    time: np.ndarray,
    candidate: TransitCandidate,
) -> np.ndarray:
    """Generate a transit model from a ``TransitCandidate``.

    Convenience wrapper around ``transit_model``.

    Args:
        time: Array of timestamps.
        candidate: A detected transit candidate.

    Returns:
        Array of model flux values.
    """
    return transit_model(
        time=time,
        period=candidate.period,
        epoch=candidate.epoch,
        duration=candidate.duration,
        depth=candidate.depth,
    )


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve at a given period and epoch.

    Returns the data sorted by phase so it can be plotted directly.

    Args:
        time: Array of timestamps.
        flux: Array of flux values.
        period: Folding period in days.
        epoch: Reference epoch (mid-transit time).

    Returns:
        Tuple of ``(phase, flux)`` arrays sorted by phase.
        Phase runs from -0.5 to +0.5, with transit centred at 0.0.
    """
    phase = ((time - epoch) % period) / period
    # Shift to [-0.5, 0.5] with transit at 0.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a phase-folded light curve for cleaner visualization.

    Args:
        phase: Phase array (from ``phase_fold``).
        flux: Flux array.
        n_bins: Number of phase bins.

    Returns:
        Tuple of ``(bin_centers, bin_means, bin_stds)``.
    """
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) >= 1:
            bin_means[i] = np.mean(flux[mask])
            bin_stds[i] = np.std(flux[mask]) if np.sum(mask) > 1 else 0.0

    # Remove empty bins
    valid = ~np.isnan(bin_means)
    return bin_centers[valid], bin_means[valid], bin_stds[valid]
