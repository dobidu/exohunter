"""Tests for the preprocessing pipeline.

All tests use synthetic data — no network access required.
Verifies that the pipeline preserves transit signals while
removing noise and artefacts.
"""

import numpy as np
import pytest
from lightkurve import LightCurve

from tests.conftest import make_synthetic_transit_lc


class TestPreprocessingSingle:
    """Test the single-target preprocessing pipeline."""

    def test_preserves_transit_signal(self, synthetic_lc: LightCurve) -> None:
        """Verify that the preprocessing pipeline preserves a transit dip.

        The pipeline should remove noise and stellar variability but
        keep the transit signal intact.
        """
        from exohunter.preprocessing.pipeline import preprocess_single

        processed = preprocess_single(synthetic_lc, tic_id="TEST_001")

        # The processed flux should still show a dip around the transit
        # Phase-fold at the known period and check depth
        period = 10.0
        phase = (processed.time % period) / period
        in_transit = np.abs(phase - 0.5) < 0.01  # narrow window around transit
        out_transit = np.abs(phase - 0.5) > 0.05

        if np.sum(in_transit) > 0 and np.sum(out_transit) > 0:
            mean_in = np.median(processed.flux[in_transit])
            mean_out = np.median(processed.flux[out_transit])
            measured_depth = mean_out - mean_in

            # The depth should be roughly preserved (within 50% of original)
            assert measured_depth > 0.005, (
                f"Transit signal too weak after preprocessing: "
                f"depth={measured_depth:.4f}, expected ~0.01"
            )

    def test_output_has_correct_fields(self, synthetic_lc: LightCurve) -> None:
        """Verify that ProcessedLightCurve contains all required fields."""
        from exohunter.preprocessing.pipeline import preprocess_single

        processed = preprocess_single(synthetic_lc, tic_id="TEST_002")

        assert len(processed.time) > 0
        assert len(processed.flux) == len(processed.time)
        assert len(processed.flux_err) == len(processed.time)
        assert processed.tic_id == "TEST_002"
        assert isinstance(processed.cdpp, float)

    def test_removes_outliers(self) -> None:
        """Verify that extreme outliers are removed."""
        from exohunter.preprocessing.clean import remove_outliers

        rng = np.random.default_rng(42)
        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000) + rng.normal(0, 0.001, 5000)

        # Inject some extreme outliers
        flux[100] = 1.5   # cosmic ray
        flux[200] = 0.5   # instrumental glitch
        flux[300] = 2.0   # very extreme

        lc = LightCurve(time=time, flux=flux)
        cleaned = remove_outliers(lc, sigma=5.0)

        # Outliers should be removed — fewer points
        assert len(cleaned) < len(lc)
        # Remaining flux should be within reasonable bounds
        assert np.all(cleaned.flux.value < 1.1)
        assert np.all(cleaned.flux.value > 0.9)

    def test_normalization(self) -> None:
        """Verify that normalized flux has median ≈ 1.0."""
        from exohunter.preprocessing.normalize import normalize_lightcurve

        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000) * 42000.0  # arbitrary instrumental units

        lc = LightCurve(time=time, flux=flux)
        normalized = normalize_lightcurve(lc)

        median_flux = float(np.median(normalized.flux.value))
        assert abs(median_flux - 1.0) < 0.01, (
            f"Median flux after normalization: {median_flux:.4f}, expected ~1.0"
        )

    def test_to_lightcurve_roundtrip(self, synthetic_lc: LightCurve) -> None:
        """Verify that ProcessedLightCurve.to_lightcurve() works."""
        from exohunter.preprocessing.pipeline import preprocess_single

        processed = preprocess_single(synthetic_lc, tic_id="TEST_RT")
        lc_back = processed.to_lightcurve()

        assert len(lc_back) == len(processed.time)
        np.testing.assert_array_almost_equal(
            lc_back.flux.value, processed.flux
        )
