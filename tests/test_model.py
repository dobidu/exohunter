"""Tests for the transit model and phase-folding utilities.

Verifies the trapezoidal transit model, phase folding, and phase
binning produce mathematically correct results.
"""

import numpy as np
import pytest

from exohunter.detection.model import (
    bin_phase_curve,
    phase_fold,
    transit_model,
    transit_model_from_candidate,
)
from tests.conftest import make_candidate


class TestTransitModel:
    """Test the trapezoidal transit model generator."""

    def test_out_of_transit_is_one(self) -> None:
        """Flux far from the transit must be exactly 1.0."""
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # far from epoch
        model = transit_model(time, period=10.0, epoch=5.0, duration=0.2, depth=0.01)

        np.testing.assert_array_equal(model, 1.0)

    def test_in_transit_depth(self) -> None:
        """The flat bottom of the transit must equal 1.0 - depth."""
        # Place a sample exactly at the transit centre
        time = np.array([5.0])  # exactly at epoch
        model = transit_model(time, period=10.0, epoch=5.0, duration=0.2, depth=0.01)

        assert abs(model[0] - 0.99) < 1e-10, f"Expected 0.99, got {model[0]}"

    def test_transit_is_periodic(self) -> None:
        """Transits must repeat at every multiple of the period."""
        time = np.array([5.0, 15.0, 25.0, 35.0])  # epoch, epoch+P, epoch+2P, epoch+3P
        model = transit_model(time, period=10.0, epoch=5.0, duration=0.2, depth=0.01)

        # All four times are at transit centre — all should be 1-depth
        expected = 1.0 - 0.01
        np.testing.assert_allclose(model, expected, atol=1e-10)

    def test_ingress_egress_interpolation(self) -> None:
        """The ingress/egress regions must interpolate linearly."""
        # With duration=1.0 and ingress_fraction=0.1, ingress lasts 0.1 days
        # Half-duration = 0.5, ingress region = [0.4, 0.5] from centre
        # At the very edge (0.5 from centre) flux should be ~1.0
        # At the start of ingress (0.4 from centre) flux should be ~1-depth
        time = np.linspace(4.0, 6.0, 1000)  # around epoch=5
        model = transit_model(
            time, period=20.0, epoch=5.0, duration=1.0,
            depth=0.05, ingress_fraction=0.1,
        )

        # Centre should be at full depth
        centre_idx = np.argmin(np.abs(time - 5.0))
        assert abs(model[centre_idx] - 0.95) < 1e-6

        # Outside transit should be 1.0
        outside_idx = np.argmin(np.abs(time - 4.0))
        assert abs(model[outside_idx] - 1.0) < 1e-6

        # Model should be symmetric around the epoch
        left_half = model[:len(model) // 2]
        right_half = model[len(model) // 2:][::-1]
        # Allow small asymmetry from discrete sampling
        np.testing.assert_allclose(left_half, right_half, atol=1e-3)

    def test_from_candidate_wrapper(self) -> None:
        """transit_model_from_candidate must give same result as transit_model."""
        candidate = make_candidate(period=10.0, epoch=5.0, duration=0.2, depth=0.01)
        time = np.linspace(0, 20, 500)

        model_direct = transit_model(time, period=10.0, epoch=5.0, duration=0.2, depth=0.01)
        model_wrapper = transit_model_from_candidate(time, candidate)

        np.testing.assert_array_equal(model_direct, model_wrapper)

    def test_model_values_bounded(self) -> None:
        """Model flux must always be between 1-depth and 1.0."""
        time = np.linspace(0, 100, 10000)
        depth = 0.03
        model = transit_model(time, period=5.0, epoch=2.5, duration=0.5, depth=depth)

        assert np.all(model >= 1.0 - depth - 1e-10)
        assert np.all(model <= 1.0 + 1e-10)


class TestPhaseFold:
    """Test the phase-folding function."""

    def test_phase_range(self) -> None:
        """Phase values must be in [-0.5, +0.5]."""
        time = np.linspace(0, 100, 5000)
        flux = np.ones_like(time)

        phase, _ = phase_fold(time, flux, period=10.0, epoch=5.0)

        assert np.all(phase >= -0.5)
        assert np.all(phase <= 0.5)

    def test_output_is_sorted(self) -> None:
        """The returned arrays must be sorted by phase."""
        time = np.linspace(0, 100, 5000)
        flux = np.ones_like(time)

        phase, _ = phase_fold(time, flux, period=10.0, epoch=5.0)

        assert np.all(np.diff(phase) >= 0), "Phase array is not sorted"

    def test_preserves_length(self) -> None:
        """Phase folding must not change the number of data points."""
        time = np.linspace(0, 50, 3000)
        flux = np.ones_like(time)

        phase, flux_out = phase_fold(time, flux, period=7.0, epoch=3.5)

        assert len(phase) == len(time)
        assert len(flux_out) == len(time)

    def test_transit_at_phase_zero(self) -> None:
        """After folding, the transit dip should be centred near phase=0."""
        from tests.conftest import make_synthetic_transit_lc

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0001, n_points=20000,
        )
        phase, flux = phase_fold(
            lc.time.value, lc.flux.value, period=10.0, epoch=5.0,
        )

        # The deepest points should be near phase=0
        near_zero = np.abs(phase) < 0.02
        far_from_zero = np.abs(phase) > 0.1

        mean_at_zero = np.median(flux[near_zero])
        mean_away = np.median(flux[far_from_zero])

        assert mean_at_zero < mean_away, (
            f"Transit not at phase zero: flux@0={mean_at_zero:.4f}, "
            f"flux@far={mean_away:.4f}"
        )


class TestBinPhaseCurve:
    """Test the phase-curve binning function."""

    def test_bin_centers_range(self) -> None:
        """Bin centers must be within [-0.5, +0.5]."""
        phase = np.linspace(-0.5, 0.5, 10000)
        flux = np.ones_like(phase)

        centers, _, _ = bin_phase_curve(phase, flux, n_bins=100)

        assert np.all(centers >= -0.5)
        assert np.all(centers <= 0.5)

    def test_bin_count_reasonable(self) -> None:
        """The number of returned bins must not exceed n_bins."""
        phase = np.linspace(-0.5, 0.5, 10000)
        flux = np.ones_like(phase)

        centers, means, stds = bin_phase_curve(phase, flux, n_bins=200)

        assert len(centers) <= 200
        assert len(means) == len(centers)
        assert len(stds) == len(centers)

    def test_preserves_mean(self) -> None:
        """For flat data, all bin means should be ≈ 1.0."""
        phase = np.linspace(-0.5, 0.5, 10000)
        flux = np.ones_like(phase)

        _, means, _ = bin_phase_curve(phase, flux, n_bins=50)

        np.testing.assert_allclose(means, 1.0, atol=1e-10)

    def test_transit_visible_in_bins(self) -> None:
        """A transit dip must be visible in the binned data near phase=0."""
        from tests.conftest import make_synthetic_transit_lc

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0001, n_points=20000,
        )
        phase, flux = phase_fold(lc.time.value, lc.flux.value, period=10.0, epoch=5.0)
        centers, means, _ = bin_phase_curve(phase, flux, n_bins=200)

        # Find bins near phase=0
        near_zero = np.abs(centers) < 0.02
        far = np.abs(centers) > 0.1

        assert np.any(near_zero), "No bins near phase zero"
        assert np.any(far), "No bins far from phase zero"

        min_at_zero = np.min(means[near_zero])
        median_far = np.median(means[far])

        assert min_at_zero < median_far - 0.003, (
            f"Transit dip not visible in bins: min@0={min_at_zero:.4f}, "
            f"median@far={median_far:.4f}"
        )
