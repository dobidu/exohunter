"""Shared test fixtures for ExoHunter tests.

All test data is synthetic — no network access required.

Fixtures are organized by scope:
    - ``synthetic_lc``, ``noisy_lc``, etc. — per-test LightCurve objects
    - ``make_synthetic_transit_lc`` — factory function for custom parameters
    - ``make_candidate`` — factory for TransitCandidate with sensible defaults
    - ``make_multi_planet_lc`` — factory for multi-planet light curves
"""

import numpy as np
import pytest
from lightkurve import LightCurve

from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult


def make_synthetic_transit_lc(
    period: float = 10.0,
    depth: float = 0.01,
    duration: float = 0.2,
    noise: float = 0.001,
    n_points: int = 10000,
    baseline_days: float = 90.0,
    seed: int = 42,
) -> LightCurve:
    """Create a synthetic light curve with injected transits.

    Args:
        period: Orbital period in days.
        depth: Fractional transit depth.
        duration: Transit duration in days.
        noise: Gaussian noise standard deviation.
        n_points: Number of data points.
        baseline_days: Total observation span in days.
        seed: Random seed for reproducibility.

    Returns:
        A ``LightCurve`` with periodic box-shaped transit dips.
    """
    rng = np.random.default_rng(seed)

    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones_like(time)

    # Inject box-shaped transits centred at phase = 0.5
    phase = (time % period) / period
    in_transit = np.abs(phase - 0.5) < (duration / period / 2)
    flux[in_transit] -= depth

    # Add Gaussian noise
    flux += rng.normal(0, noise, n_points)

    return LightCurve(time=time, flux=flux)


def make_multi_planet_lc(
    planets: list[dict],
    noise: float = 0.0003,
    n_points: int = 25000,
    baseline_days: float = 351.0,
    seed: int = 42,
) -> LightCurve:
    """Create a synthetic light curve with multiple transiting planets.

    Args:
        planets: List of dicts, each with ``period``, ``depth``,
            ``duration``, and ``epoch`` keys.
        noise: Gaussian noise standard deviation.
        n_points: Number of data points.
        baseline_days: Total observation span.
        seed: Random seed.

    Returns:
        A ``LightCurve`` with all planets' transits superimposed.
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones(n_points, dtype=np.float64)

    for planet in planets:
        period = planet["period"]
        epoch = planet.get("epoch", period * 0.5)
        depth = planet["depth"]
        duration = planet["duration"]
        half_dur = duration / 2.0

        phase_time = ((time - epoch + period / 2) % period) - period / 2
        in_transit = np.abs(phase_time) < half_dur
        flux[in_transit] -= depth

    flux += rng.normal(0, noise, n_points)
    return LightCurve(time=time, flux=flux)


def make_candidate(**kwargs) -> TransitCandidate:
    """Create a TransitCandidate with sensible defaults.

    Any keyword argument overrides the default value.
    """
    defaults = {
        "tic_id": "TIC 999999999",
        "period": 10.0,
        "epoch": 5.0,
        "duration": 0.2,
        "depth": 0.01,
        "snr": 15.0,
        "bls_power": 0.5,
        "n_transits": 9,
        "name": "",
    }
    defaults.update(kwargs)
    return TransitCandidate(**defaults)


def make_validation(
    is_valid: bool = True,
    v_shape_pass: bool = True,
) -> ValidationResult:
    """Create a ValidationResult with configurable V-shape test."""
    return ValidationResult(
        is_valid=is_valid,
        tests={"v_shape": v_shape_pass, "snr": True, "depth": True,
               "duration": True, "n_transits": True, "harmonic": True},
        flags=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def toi700_lc() -> LightCurve:
    """A synthetic TOI-700-like light curve with 3 planets."""
    return make_multi_planet_lc(
        planets=[
            {"period": 9.977,  "epoch": 2.5, "depth": 0.00058, "duration": 0.095},
            {"period": 16.051, "epoch": 4.1, "depth": 0.00078, "duration": 0.125},
            {"period": 37.426, "epoch": 8.7, "depth": 0.00081, "duration": 0.148},
        ],
        noise=0.0003,
    )
