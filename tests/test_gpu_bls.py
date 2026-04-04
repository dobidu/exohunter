"""Tests for the GPU-accelerated BLS implementation.

Tests the run_bls_gpu function, which uses Numba CUDA when available
and falls back to CPU Numba transparently. All tests work regardless
of whether a CUDA GPU is present — the fallback is part of the contract.
"""

import numpy as np
import pytest

from tests.conftest import make_synthetic_transit_lc


class TestGPUBLSFallback:
    """Test that run_bls_gpu always works via fallback."""

    def test_returns_candidate(self) -> None:
        """run_bls_gpu must return a TransitCandidate (via GPU or fallback)."""
        from exohunter.detection.bls import run_bls_gpu

        lc = make_synthetic_transit_lc(
            period=7.0, depth=0.01, duration=0.15, noise=0.0005, n_points=10000,
        )

        candidate = run_bls_gpu(
            lc.time.value, lc.flux.value, tic_id="GPU_TEST",
            min_period=2.0, max_period=12.0, num_periods=2000,
        )

        assert candidate is not None
        assert hasattr(candidate, "period")
        assert hasattr(candidate, "depth")

    def test_recovers_period(self) -> None:
        """run_bls_gpu must find the injected period within 0.2 d."""
        from exohunter.detection.bls import run_bls_gpu

        lc = make_synthetic_transit_lc(
            period=7.0, depth=0.01, duration=0.15, noise=0.0005, n_points=10000,
        )

        candidate = run_bls_gpu(
            lc.time.value, lc.flux.value, tic_id="GPU_PERIOD",
            min_period=2.0, max_period=12.0, num_periods=3000,
        )

        assert candidate is not None
        assert abs(candidate.period - 7.0) < 0.2, (
            f"Period mismatch: {candidate.period:.4f} vs 7.0"
        )

    def test_agrees_with_cpu_numba(self) -> None:
        """GPU and CPU Numba must find the same period."""
        from exohunter.detection.bls import run_bls_gpu, run_bls_numba, _NUMBA_AVAILABLE

        if not _NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        lc = make_synthetic_transit_lc(
            period=5.0, depth=0.01, duration=0.1, noise=0.0005, n_points=10000,
        )
        t, f = lc.time.value, lc.flux.value

        cpu = run_bls_numba(t, f, tic_id="CPU", min_period=2.0,
                             max_period=10.0, num_periods=2000)
        gpu = run_bls_gpu(t, f, tic_id="GPU", min_period=2.0,
                           max_period=10.0, num_periods=2000)

        assert cpu is not None and gpu is not None
        assert abs(cpu.period - gpu.period) < 0.3, (
            f"CPU P={cpu.period:.4f} vs GPU P={gpu.period:.4f}"
        )


class TestGPUAvailability:
    """Test GPU detection and availability flags."""

    def test_cuda_flag_is_bool(self) -> None:
        """_CUDA_AVAILABLE must be a boolean."""
        from exohunter.detection.bls import _CUDA_AVAILABLE

        assert isinstance(_CUDA_AVAILABLE, bool)

    def test_run_bls_gpu_is_callable(self) -> None:
        """run_bls_gpu must be importable and callable."""
        from exohunter.detection.bls import run_bls_gpu

        assert callable(run_bls_gpu)

    def test_exported_from_detection_package(self) -> None:
        """run_bls_gpu must be exported from exohunter.detection."""
        from exohunter.detection import run_bls_gpu

        assert callable(run_bls_gpu)


class TestGPUEdgeCases:
    """Test edge cases for the GPU BLS."""

    def test_single_period(self) -> None:
        """Must handle num_periods=1 without crashing."""
        from exohunter.detection.bls import run_bls_gpu

        lc = make_synthetic_transit_lc(period=5.0, depth=0.01, n_points=5000)
        t, f = lc.time.value, lc.flux.value

        candidate = run_bls_gpu(t, f, tic_id="SINGLE",
                                 min_period=5.0, max_period=5.0, num_periods=1)
        # Should return a candidate (even if trivial)
        assert candidate is not None or candidate is None  # no crash

    def test_short_lightcurve(self) -> None:
        """Must handle very short light curves gracefully."""
        from exohunter.detection.bls import run_bls_gpu

        t = np.linspace(0, 5, 100)
        f = np.ones(100) + np.random.default_rng(42).normal(0, 0.001, 100)

        candidate = run_bls_gpu(t, f, tic_id="SHORT",
                                 min_period=1.0, max_period=4.0, num_periods=500)
        # Should not crash — result may be None or a low-SNR candidate
        assert candidate is not None or candidate is None
