"""Shared test fixtures for ExoHunter tests.

All test data is synthetic — no network access required.
"""

import numpy as np
import pytest
from lightkurve import LightCurve


def make_synthetic_transit_lc(
    period: float = 10.0,
    depth: float = 0.01,
    duration: float = 0.2,
    noise: float = 0.001,
    n_points: int = 10000,
    baseline_days: float = 90.0,
) -> LightCurve:
    """Create a synthetic light curve with injected transits.

    Args:
        period: Orbital period in days.
        depth: Fractional transit depth.
        duration: Transit duration in days.
        noise: Gaussian noise standard deviation.
        n_points: Number of data points.
        baseline_days: Total observation span in days.

    Returns:
        A ``LightCurve`` with periodic box-shaped transit dips.
    """
    rng = np.random.default_rng(42)

    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones_like(time)

    # Inject box-shaped transits centred at phase = 0.5
    phase = (time % period) / period
    in_transit = np.abs(phase - 0.5) < (duration / period / 2)
    flux[in_transit] -= depth

    # Add Gaussian noise
    flux += rng.normal(0, noise, n_points)

    return LightCurve(time=time, flux=flux)


@pytest.fixture
def synthetic_lc() -> LightCurve:
    """A synthetic light curve with a 10-day, 1% transit."""
    return make_synthetic_transit_lc(
        period=10.0, depth=0.01, duration=0.2, noise=0.001
    )


@pytest.fixture
def shallow_transit_lc() -> LightCurve:
    """A synthetic light curve with a very shallow transit (hard to detect)."""
    return make_synthetic_transit_lc(
        period=5.0, depth=0.0005, duration=0.15, noise=0.001
    )


@pytest.fixture
def deep_transit_lc() -> LightCurve:
    """A synthetic light curve with a deep transit (eclipsing binary)."""
    return make_synthetic_transit_lc(
        period=3.0, depth=0.10, duration=0.3, noise=0.001
    )


@pytest.fixture
def noisy_lc() -> LightCurve:
    """A flat synthetic light curve with no transit (pure noise)."""
    rng = np.random.default_rng(123)
    time = np.linspace(0, 90, 10000)
    flux = 1.0 + rng.normal(0, 0.002, 10000)
    return LightCurve(time=time, flux=flux)
