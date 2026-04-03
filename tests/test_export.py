"""Tests for the catalog export module (CSV, FITS, VOTable).

Verifies that each export format produces a valid file that can be
read back and contains the correct data.
"""

import numpy as np
import pytest
from pathlib import Path

from exohunter.catalog.candidates import CandidateCatalog
from exohunter.catalog.export import export_to_csv, export_to_fits, export_to_votable
from tests.conftest import make_candidate, make_validation


@pytest.fixture
def sample_catalog() -> CandidateCatalog:
    """A catalog with two candidates for export testing."""
    catalog = CandidateCatalog()
    catalog.add(
        make_candidate(tic_id="TIC 111", period=5.0, depth=0.01, snr=15.0),
        make_validation(),
    )
    catalog.add(
        make_candidate(tic_id="TIC 222", period=10.0, depth=0.005, snr=8.0),
        make_validation(v_shape_pass=False),
    )
    return catalog


class TestExportCSV:
    """Test CSV export."""

    def test_creates_file(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """export_to_csv must create a file that exists."""
        path = export_to_csv(sample_catalog, tmp_path / "test.csv")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip_row_count(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """CSV must contain one row per candidate."""
        import pandas as pd

        path = export_to_csv(sample_catalog, tmp_path / "test.csv")
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_contains_score_column(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """CSV must include the score column."""
        import pandas as pd

        path = export_to_csv(sample_catalog, tmp_path / "test.csv")
        df = pd.read_csv(path)
        assert "score" in df.columns

    def test_default_path(self, sample_catalog: CandidateCatalog) -> None:
        """export_to_csv with no path must use the default output dir."""
        path = export_to_csv(sample_catalog)
        assert "candidates.csv" in path.name
        # Clean up
        path.unlink(missing_ok=True)


class TestExportFITS:
    """Test FITS export."""

    def test_creates_file(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """export_to_fits must create a file."""
        path = export_to_fits(sample_catalog, tmp_path / "test.fits")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip_row_count(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """FITS table must contain the correct number of rows."""
        from astropy.table import Table

        path = export_to_fits(sample_catalog, tmp_path / "test.fits")
        table = Table.read(path, format="fits")
        assert len(table) == 2

    def test_preserves_period(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """Period values must survive the FITS roundtrip."""
        from astropy.table import Table

        path = export_to_fits(sample_catalog, tmp_path / "test.fits")
        table = Table.read(path, format="fits")
        periods = sorted(table["period"].data)
        assert abs(periods[0] - 5.0) < 0.001
        assert abs(periods[1] - 10.0) < 0.001


class TestExportVOTable:
    """Test VOTable export."""

    def test_creates_file(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """export_to_votable must create a file."""
        path = export_to_votable(sample_catalog, tmp_path / "test.votable.xml")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_is_valid_xml(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """The output must be parseable as VOTable XML."""
        from astropy.table import Table

        path = export_to_votable(sample_catalog, tmp_path / "test.votable.xml")
        table = Table.read(path, format="votable")
        assert len(table) == 2

    def test_has_ucd_metadata(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """VOTable columns must have UCD metadata for VO tools."""
        from astropy.table import Table

        path = export_to_votable(sample_catalog, tmp_path / "test.votable.xml")
        table = Table.read(path, format="votable")

        # Period should have UCD "time.period"
        assert table["period"].meta.get("ucd") == "time.period"
        # SNR should have UCD "stat.snr"
        assert table["snr"].meta.get("ucd") == "stat.snr"

    def test_has_units(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """Period and duration columns must have unit 'd' (days)."""
        from astropy.table import Table

        path = export_to_votable(sample_catalog, tmp_path / "test.votable.xml")
        table = Table.read(path, format="votable")

        assert str(table["period"].unit) == "d"
        assert str(table["duration"].unit) == "d"

    def test_preserves_data(self, sample_catalog: CandidateCatalog, tmp_path: Path) -> None:
        """Candidate data must survive the VOTable roundtrip."""
        from astropy.table import Table

        path = export_to_votable(sample_catalog, tmp_path / "test.votable.xml")
        table = Table.read(path, format="votable")

        periods = sorted(table["period"].data)
        assert abs(periods[0] - 5.0) < 0.001
        assert abs(periods[1] - 10.0) < 0.001

    def test_empty_catalog(self, tmp_path: Path) -> None:
        """Exporting an empty catalog must not crash."""
        catalog = CandidateCatalog()
        path = export_to_votable(catalog, tmp_path / "empty.votable.xml")
        assert path.exists()
