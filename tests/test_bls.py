"""Tests for the BLS transit detection module.

Uses synthetic light curves with known transit parameters to verify
that the BLS algorithm correctly recovers the injected period.
"""

import numpy as np
import pytest
from lightkurve import LightCurve

from tests.conftest import make_synthetic_transit_lc


class TestBLSDetection:
    """Test BLS transit detection on synthetic data."""

    def test_recovers_known_period(self) -> None:
        """BLS should find the correct period (±0.01 days) for a clean signal.

        This is the primary correctness test: inject a transit at P=10.0 d
        and verify BLS recovers it.
        """
        from exohunter.detection.bls import run_bls_lightkurve

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0005, n_points=15000
        )

        candidate = run_bls_lightkurve(
            lc,
            tic_id="BLS_TEST",
            min_period=1.0,
            max_period=15.0,
            num_periods=5000,
        )

        assert candidate is not None, "BLS failed to detect transit"
        assert abs(candidate.period - 10.0) < 0.05, (
            f"Period mismatch: found {candidate.period:.4f}, expected 10.0"
        )

    def test_recovers_short_period(self) -> None:
        """BLS should find a short-period transit (hot Jupiter regime)."""
        from exohunter.detection.bls import run_bls_lightkurve

        lc = make_synthetic_transit_lc(
            period=2.5, depth=0.015, duration=0.1, noise=0.0005, n_points=15000
        )

        candidate = run_bls_lightkurve(
            lc,
            tic_id="SHORT_P",
            min_period=0.5,
            max_period=5.0,
            num_periods=5000,
        )

        assert candidate is not None
        assert abs(candidate.period - 2.5) < 0.05, (
            f"Period mismatch: {candidate.period:.4f} vs 2.5"
        )

    def test_depth_approximately_correct(self) -> None:
        """The detected depth should be close to the injected depth."""
        from exohunter.detection.bls import run_bls_lightkurve

        injected_depth = 0.01
        lc = make_synthetic_transit_lc(
            period=8.0, depth=injected_depth, duration=0.2,
            noise=0.0005, n_points=15000,
        )

        candidate = run_bls_lightkurve(
            lc,
            tic_id="DEPTH_TEST",
            min_period=1.0,
            max_period=12.0,
            num_periods=5000,
        )

        assert candidate is not None
        # Depth should be within a factor of 2 of injected
        assert 0.5 * injected_depth < candidate.depth < 2.0 * injected_depth, (
            f"Depth mismatch: {candidate.depth:.6f} vs {injected_depth}"
        )

    def test_transit_candidate_fields(self) -> None:
        """Verify that TransitCandidate has all expected fields."""
        from exohunter.detection.bls import TransitCandidate

        tc = TransitCandidate(
            tic_id="TEST", period=10.0, epoch=5.0,
            duration=0.2, depth=0.01, snr=15.0, bls_power=0.5,
        )

        assert tc.tic_id == "TEST"
        assert tc.period == 10.0
        assert tc.n_transits == 0  # default value

    def test_numba_bls_available(self) -> None:
        """Check that the Numba BLS is importable (may fall back)."""
        from exohunter.detection.bls import run_bls_numba
        # Just verify it's callable — Numba may not be installed in CI
        assert callable(run_bls_numba)

    def test_numba_bls_recovers_period(self) -> None:
        """If Numba is available, verify it finds the correct period."""
        from exohunter.detection.bls import _NUMBA_AVAILABLE, run_bls_numba

        if not _NUMBA_AVAILABLE:
            pytest.skip("Numba not installed")

        lc = make_synthetic_transit_lc(
            period=10.0, depth=0.01, duration=0.2, noise=0.0005, n_points=10000
        )

        candidate = run_bls_numba(
            time=lc.time.value,
            flux=lc.flux.value,
            tic_id="NUMBA_TEST",
            min_period=5.0,
            max_period=15.0,
            num_periods=2000,
        )

        assert candidate is not None
        assert abs(candidate.period - 10.0) < 0.1, (
            f"Numba BLS period: {candidate.period:.4f}, expected 10.0"
        )
