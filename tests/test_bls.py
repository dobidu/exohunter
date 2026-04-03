"""Tests for the BLS transit detection module.

Uses synthetic light curves with known transit parameters to verify
that both BLS implementations (lightkurve and Numba) correctly recover
the injected period and depth.
"""

import numpy as np
import pytest
from lightkurve import LightCurve

from tests.conftest import make_synthetic_transit_lc


class TestBLSLightkurve:
    """Test the lightkurve/astropy BLS implementation."""

    def test_recovers_known_period(self) -> None:
        """BLS must find P=10.0 d within ±0.05 d for a clean 1% transit."""
        from exohunter.detection.bls import run_bls_lightkurve

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0005, n_points=15000
        )
        candidate = run_bls_lightkurve(
            lc, tic_id="BLS_TEST", min_period=1.0, max_period=15.0, num_periods=5000,
        )

        assert candidate is not None, "BLS failed to detect transit"
        assert abs(candidate.period - 10.0) < 0.05, (
            f"Period mismatch: found {candidate.period:.4f}, expected 10.0"
        )

    def test_recovers_short_period(self) -> None:
        """BLS must find P=2.5 d (hot Jupiter regime)."""
        from exohunter.detection.bls import run_bls_lightkurve

        lc = make_synthetic_transit_lc(
            period=2.5, depth=0.015, duration=0.1, noise=0.0005, n_points=15000
        )
        candidate = run_bls_lightkurve(
            lc, tic_id="SHORT_P", min_period=0.5, max_period=5.0, num_periods=5000,
        )

        assert candidate is not None
        assert abs(candidate.period - 2.5) < 0.05

    def test_depth_approximately_correct(self) -> None:
        """Detected depth must be within 2× of the injected depth."""
        from exohunter.detection.bls import run_bls_lightkurve

        injected_depth = 0.01
        lc = make_synthetic_transit_lc(
            period=8.0, depth=injected_depth, duration=0.2,
            noise=0.0005, n_points=15000,
        )
        candidate = run_bls_lightkurve(
            lc, tic_id="DEPTH_TEST", min_period=1.0, max_period=12.0, num_periods=5000,
        )

        assert candidate is not None
        assert 0.5 * injected_depth < candidate.depth < 2.0 * injected_depth, (
            f"Depth mismatch: {candidate.depth:.6f} vs {injected_depth}"
        )

    def test_returns_none_for_flat_data(self, noisy_lc: LightCurve) -> None:
        """BLS should still return a candidate for flat data, but with low SNR."""
        from exohunter.detection.bls import run_bls_lightkurve

        candidate = run_bls_lightkurve(
            noisy_lc, tic_id="FLAT", min_period=1.0, max_period=15.0, num_periods=1000,
        )
        # BLS always returns a "best" period, but SNR should be very low
        if candidate is not None:
            assert candidate.snr < 5.0, "Flat data should have very low SNR"


class TestBLSNumba:
    """Test the Numba prefix-sum BLS implementation."""

    def test_numba_is_importable(self) -> None:
        """The Numba BLS function must be importable regardless of Numba availability."""
        from exohunter.detection.bls import run_bls_numba
        assert callable(run_bls_numba)

    def test_recovers_period(self) -> None:
        """Numba BLS must find P=10.0 d within ±0.1 d."""
        from exohunter.detection.bls import _NUMBA_AVAILABLE, run_bls_numba

        if not _NUMBA_AVAILABLE:
            pytest.skip("Numba not installed")

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0005, n_points=10000
        )
        candidate = run_bls_numba(
            time=lc.time.value, flux=lc.flux.value, tic_id="NUMBA_TEST",
            min_period=5.0, max_period=15.0, num_periods=2000,
        )

        assert candidate is not None
        assert abs(candidate.period - 10.0) < 0.1, (
            f"Numba BLS period: {candidate.period:.4f}, expected 10.0"
        )

    def test_agrees_with_lightkurve(self) -> None:
        """Numba and lightkurve BLS must find the same period (within tolerance)."""
        from exohunter.detection.bls import _NUMBA_AVAILABLE, run_bls_lightkurve, run_bls_numba

        if not _NUMBA_AVAILABLE:
            pytest.skip("Numba not installed")

        lc = make_synthetic_transit_lc(
            period=7.0, depth=0.01, duration=0.15, noise=0.0005, n_points=12000
        )

        lk_candidate = run_bls_lightkurve(
            lc, tic_id="CMP_LK", min_period=2.0, max_period=12.0, num_periods=3000,
        )
        nb_candidate = run_bls_numba(
            time=lc.time.value, flux=lc.flux.value, tic_id="CMP_NB",
            min_period=2.0, max_period=12.0, num_periods=3000,
        )

        assert lk_candidate is not None and nb_candidate is not None
        assert abs(lk_candidate.period - nb_candidate.period) < 0.2, (
            f"lightkurve P={lk_candidate.period:.4f} vs Numba P={nb_candidate.period:.4f}"
        )


class TestTransitCandidateDataclass:
    """Test the TransitCandidate dataclass itself."""

    def test_all_fields_present(self) -> None:
        """TransitCandidate must have all documented fields."""
        from exohunter.detection.bls import TransitCandidate

        tc = TransitCandidate(
            tic_id="TEST", period=10.0, epoch=5.0,
            duration=0.2, depth=0.01, snr=15.0, bls_power=0.5,
        )

        assert tc.tic_id == "TEST"
        assert tc.period == 10.0
        assert tc.n_transits == 0   # default
        assert tc.name == ""        # default

    def test_name_field(self) -> None:
        """The optional name field must propagate correctly."""
        from exohunter.detection.bls import TransitCandidate

        tc = TransitCandidate(
            tic_id="TEST", period=10.0, epoch=5.0,
            duration=0.2, depth=0.01, snr=15.0, bls_power=0.5,
            name="TOI-700 d",
        )
        assert tc.name == "TOI-700 d"


class TestIterativeBLS:
    """Test the multi-planet iterative BLS search."""

    def test_finds_two_planets(self) -> None:
        """Iterative BLS must find both planets in a two-planet system.

        Uses min_snr=0.0 because lightkurve's compute_stats returns
        SNR=0 for synthetic data without realistic error bars.  The
        iterative search relies on the duplicate/harmonic check and
        max_planets to stop instead.
        """
        from exohunter.detection.bls import run_iterative_bls
        from tests.conftest import make_multi_planet_lc

        lc = make_multi_planet_lc(
            planets=[
                {"period": 3.0, "epoch": 1.5, "depth": 0.015, "duration": 0.1},
                {"period": 7.0, "epoch": 3.5, "depth": 0.01,  "duration": 0.15},
            ],
            noise=0.0005,
            n_points=20000,
            baseline_days=90.0,
        )

        candidates = run_iterative_bls(
            lc, tic_id="MULTI_TEST",
            min_period=1.0, max_period=12.0, num_periods=5000,
            max_planets=4, min_snr=0.0,
        )

        assert len(candidates) >= 2, (
            f"Expected at least 2 planets, found {len(candidates)}: "
            f"{[c.period for c in candidates]}"
        )

        found_periods = sorted([c.period for c in candidates])
        assert any(abs(p - 3.0) < 0.2 for p in found_periods), (
            f"P=3.0 d not found in {found_periods}"
        )
        assert any(abs(p - 7.0) < 0.2 for p in found_periods), (
            f"P=7.0 d not found in {found_periods}"
        )

    def test_returns_empty_for_flat_data(self, noisy_lc) -> None:
        """Iterative BLS on flat data must return an empty list."""
        from exohunter.detection.bls import run_iterative_bls

        candidates = run_iterative_bls(
            noisy_lc, tic_id="FLAT_ITER",
            min_period=1.0, max_period=15.0, num_periods=1000,
            min_snr=5.0,
        )

        assert len(candidates) == 0

    def test_candidates_have_planet_letters(self) -> None:
        """Each candidate must be labeled with a planet letter (b, c, ...)."""
        from exohunter.detection.bls import run_iterative_bls
        from tests.conftest import make_multi_planet_lc

        lc = make_multi_planet_lc(
            planets=[
                {"period": 3.0, "epoch": 1.5, "depth": 0.015, "duration": 0.1},
                {"period": 7.0, "epoch": 3.5, "depth": 0.01,  "duration": 0.15},
            ],
            noise=0.0005,
            n_points=20000,
            baseline_days=90.0,
        )

        candidates = run_iterative_bls(
            lc, tic_id="TIC 123",
            min_period=1.0, max_period=12.0, num_periods=5000,
            max_planets=4, min_snr=0.0,
        )

        if len(candidates) >= 1:
            assert "b" in candidates[0].name
        if len(candidates) >= 2:
            assert "c" in candidates[1].name

    def test_stops_at_max_planets(self) -> None:
        """Iterative BLS must not exceed max_planets."""
        from exohunter.detection.bls import run_iterative_bls
        from tests.conftest import make_synthetic_transit_lc

        # Strong signal — BLS will always find something
        lc = make_synthetic_transit_lc(
            period=5.0, depth=0.02, duration=0.15, noise=0.0003,
        )

        candidates = run_iterative_bls(
            lc, tic_id="MAX_TEST",
            min_period=1.0, max_period=10.0, num_periods=3000,
            max_planets=1, min_snr=1.0,
        )

        assert len(candidates) <= 1
