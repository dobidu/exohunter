"""Tests for the light curve FITS cache.

Verifies that LightCurve objects survive a write→read roundtrip
through the astropy Table-based cache, and that cache misses
are handled gracefully.
"""

import numpy as np
import pytest
from lightkurve import LightCurve
from pathlib import Path

from exohunter.ingestion.cache import (
    _tic_to_filename,
    load_from_cache,
    save_to_cache,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for cache tests."""
    return tmp_path / "test_cache"


class TestTicToFilename:
    """Test the TIC ID → filename conversion."""

    def test_standard_format(self) -> None:
        assert _tic_to_filename("TIC 150428135") == "TIC_150428135.fits"

    def test_no_space(self) -> None:
        assert _tic_to_filename("TIC150428135") == "TIC150428135.fits"

    def test_extra_whitespace(self) -> None:
        assert _tic_to_filename("  TIC 150428135  ") == "TIC_150428135.fits"

    def test_special_characters_stripped(self) -> None:
        # Any non-alphanumeric characters become underscores
        result = _tic_to_filename("TIC-150/428.135")
        assert ".fits" in result
        assert "/" not in result
        assert "-" not in result


class TestCacheRoundtrip:
    """Test write→read roundtrip of LightCurve objects."""

    def test_roundtrip_preserves_time_and_flux(self, cache_dir: Path) -> None:
        """Time and flux arrays must be identical after roundtrip."""
        rng = np.random.default_rng(42)
        time = np.linspace(0, 90, 5000)
        flux = 1.0 + rng.normal(0, 0.001, 5000)
        lc = LightCurve(time=time, flux=flux)

        save_to_cache(lc, "TIC 123456789", cache_dir)
        loaded = load_from_cache("TIC 123456789", cache_dir)

        assert loaded is not None, "Cache load returned None"
        assert len(loaded) == len(lc)
        np.testing.assert_allclose(loaded.time.value, time, atol=1e-10)
        np.testing.assert_allclose(loaded.flux.value, flux, atol=1e-10)

    def test_roundtrip_preserves_flux_err(self, cache_dir: Path) -> None:
        """Flux error array must survive the roundtrip."""
        time = np.linspace(0, 30, 1000)
        flux = np.ones(1000)
        flux_err = np.full(1000, 0.001)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)

        save_to_cache(lc, "TIC 111111111", cache_dir)
        loaded = load_from_cache("TIC 111111111", cache_dir)

        assert loaded is not None
        assert loaded.flux_err is not None
        np.testing.assert_allclose(loaded.flux_err.value, flux_err, atol=1e-10)

    def test_roundtrip_without_flux_err(self, cache_dir: Path) -> None:
        """LightCurves without flux_err must still roundtrip."""
        time = np.linspace(0, 10, 500)
        flux = np.ones(500)
        lc = LightCurve(time=time, flux=flux)

        save_to_cache(lc, "TIC 222222222", cache_dir)
        loaded = load_from_cache("TIC 222222222", cache_dir)

        assert loaded is not None
        assert len(loaded) == 500

    def test_roundtrip_large_lightcurve(self, cache_dir: Path) -> None:
        """A 20k-cadence light curve (typical TESS sector) must roundtrip."""
        rng = np.random.default_rng(99)
        time = np.linspace(0, 27.4, 20000)
        flux = 1.0 + rng.normal(0, 0.0003, 20000)
        lc = LightCurve(time=time, flux=flux)

        save_to_cache(lc, "TIC 333333333", cache_dir)
        loaded = load_from_cache("TIC 333333333", cache_dir)

        assert loaded is not None
        assert len(loaded) == 20000
        np.testing.assert_allclose(loaded.flux.value, flux, atol=1e-10)


class TestCacheMiss:
    """Test cache miss behaviour."""

    def test_missing_file_returns_none(self, cache_dir: Path) -> None:
        """load_from_cache must return None if the file doesn't exist."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = load_from_cache("TIC 000000000", cache_dir)
        assert result is None

    def test_corrupt_file_returns_none(self, cache_dir: Path) -> None:
        """A corrupt FITS file must return None (not raise)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = cache_dir / "TIC_999999999.fits"
        corrupt_path.write_text("this is not a FITS file")

        result = load_from_cache("TIC 999999999", cache_dir)
        assert result is None

    def test_overwrite_existing(self, cache_dir: Path) -> None:
        """Saving to an existing cache key must overwrite cleanly."""
        time = np.linspace(0, 10, 100)
        lc1 = LightCurve(time=time, flux=np.ones(100) * 1.0)
        lc2 = LightCurve(time=time, flux=np.ones(100) * 2.0)

        save_to_cache(lc1, "TIC 444444444", cache_dir)
        save_to_cache(lc2, "TIC 444444444", cache_dir)

        loaded = load_from_cache("TIC 444444444", cache_dir)
        assert loaded is not None
        np.testing.assert_allclose(loaded.flux.value, 2.0, atol=1e-10)
