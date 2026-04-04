"""Tests for the dashboard data overview module.

Verifies directory scanning, metadata extraction, and
base64 encoding for the Data Overview section.
"""

import json
from pathlib import Path

import numpy as np
import pytest


class TestCacheStats:
    """Test the cache scanner."""

    def test_returns_dict_with_required_keys(self) -> None:
        """scan_cache_stats must return the expected structure."""
        from exohunter.dashboard.overview import scan_cache_stats

        stats = scan_cache_stats()
        assert "n_files" in stats
        assert "total_size_mb" in stats
        assert "largest_files" in stats
        assert isinstance(stats["n_files"], int)

    def test_counts_existing_cache(self) -> None:
        """If cache files exist, count must be > 0."""
        from exohunter.dashboard.overview import scan_cache_stats
        from exohunter import config

        stats = scan_cache_stats()
        cache_files = list(config.CACHE_DIR.glob("*.fits"))
        assert stats["n_files"] == len(cache_files)


class TestBatchResults:
    """Test the batch results scanner."""

    def test_returns_list(self) -> None:
        """scan_batch_results must return a list."""
        from exohunter.dashboard.overview import scan_batch_results

        results = scan_batch_results()
        assert isinstance(results, list)

    def test_parses_existing_csv(self) -> None:
        """If sector CSVs exist, they must be parsed correctly."""
        from exohunter.dashboard.overview import scan_batch_results
        from exohunter import config

        csv_files = list(config.RESULTS_DIR.glob("sector_*.csv"))
        csv_files = [f for f in csv_files if "_candidates" not in f.stem]
        results = scan_batch_results()

        assert len(results) == len(csv_files)
        if results:
            r = results[0]
            assert "sector" in r
            assert "n_targets" in r
            assert "date" in r


class TestReportsScanner:
    """Test the reports gallery scanner."""

    def test_returns_list(self) -> None:
        """scan_reports must return a list."""
        from exohunter.dashboard.overview import scan_reports

        reports = scan_reports()
        assert isinstance(reports, list)

    def test_base64_encoding(self, tmp_path: Path) -> None:
        """load_report_as_base64 must produce a data URI."""
        from exohunter.dashboard.overview import load_report_as_base64

        # Create a tiny valid PNG (1x1 pixel)
        import struct, zlib
        def _mini_png() -> bytes:
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc
            raw = zlib.compress(b'\x00\x00\x00\x00')
            idat = struct.pack('>I', len(raw)) + b'IDAT' + raw
            idat += struct.pack('>I', zlib.crc32(b'IDAT' + raw) & 0xffffffff)
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
            return sig + ihdr + idat + iend

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_mini_png())

        result = load_report_as_base64(str(png_path))
        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_missing_file_returns_none(self) -> None:
        """load_report_as_base64 for missing file must return None."""
        from exohunter.dashboard.overview import load_report_as_base64

        result = load_report_as_base64("/nonexistent/path.png")
        assert result is None


class TestAlertsScanner:
    """Test the alerts feed scanner."""

    def test_returns_list(self) -> None:
        """scan_alerts must return a list."""
        from exohunter.dashboard.overview import scan_alerts

        alerts = scan_alerts()
        assert isinstance(alerts, list)

    def test_parses_alert_json(self, tmp_path: Path) -> None:
        """scan_alerts must parse valid JSON alert files."""
        import exohunter.config as cfg
        original = cfg.ALERTS_DIR
        cfg.ALERTS_DIR = tmp_path

        try:
            alert = {
                "alert_type": "new_candidates",
                "timestamp": "2026-04-04T12:00:00",
                "sector": 56,
                "n_candidates": 2,
                "candidates": [
                    {"tic_id": "TIC 111"},
                    {"tic_id": "TIC 222"},
                ],
            }
            (tmp_path / "alert_sector_56_20260404_120000.json").write_text(
                json.dumps(alert)
            )

            from exohunter.dashboard.overview import scan_alerts
            results = scan_alerts()

            assert len(results) == 1
            assert results[0]["n_candidates"] == 2
            assert results[0]["sector"] == 56
        finally:
            cfg.ALERTS_DIR = original


class TestMLStatus:
    """Test the ML model status scanner."""

    def test_returns_dict(self) -> None:
        """scan_ml_status must return expected keys."""
        from exohunter.dashboard.overview import scan_ml_status

        status = scan_ml_status()
        assert "rf_available" in status
        assert "cnn_available" in status
        assert isinstance(status["rf_available"], bool)
        assert isinstance(status["cnn_available"], bool)


class TestDashboardOverviewComponents:
    """Test that the overview components are present in the dashboard layout."""

    def test_layout_has_overview_ids(self) -> None:
        """The layout must contain all overview component IDs."""
        from exohunter.dashboard.app import create_app

        app = create_app()

        # Walk layout to find IDs
        def _collect_ids(component) -> set:
            ids = set()
            comp_id = getattr(component, "id", None)
            if comp_id and isinstance(comp_id, str):
                ids.add(comp_id)
            children = getattr(component, "children", None)
            if children:
                if isinstance(children, (list, tuple)):
                    for child in children:
                        ids.update(_collect_ids(child))
                elif hasattr(children, "id") or hasattr(children, "children"):
                    ids.update(_collect_ids(children))
            return ids

        all_ids = _collect_ids(app.layout)

        required = [
            "cache-stats-body",
            "ml-status-body",
            "batch-results-body",
            "reports-gallery-body",
            "alerts-feed-body",
            "report-modal",
        ]
        for comp_id in required:
            assert comp_id in all_ids, f"Missing overview component: {comp_id}"
