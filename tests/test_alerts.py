"""Tests for the automatic alert system.

Verifies alert filtering, payload construction, file output,
and webhook dispatch (mocked).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exohunter.alerts import (
    _build_alert_payload,
    check_and_dispatch_alerts,
    find_alertable_candidates,
    save_alert_file,
)


def _make_summary_df(
    n_new: int = 2,
    n_known: int = 1,
    n_low_snr: int = 1,
) -> pd.DataFrame:
    """Build a mock batch summary DataFrame."""
    rows = []
    for i in range(n_new):
        rows.append({
            "tic_id": f"TIC {900000000 + i}",
            "status": "new_candidate",
            "xmatch_class": "NEW_CANDIDATE",
            "period": 5.0 + i,
            "depth": 0.005 + i * 0.001,
            "snr": 10.0 + i * 2,
            "is_valid": True,
        })
    for i in range(n_known):
        rows.append({
            "tic_id": f"TIC {800000000 + i}",
            "status": "known_match",
            "xmatch_class": "KNOWN_MATCH",
            "period": 3.0,
            "depth": 0.01,
            "snr": 15.0,
            "is_valid": True,
        })
    for i in range(n_low_snr):
        rows.append({
            "tic_id": f"TIC {700000000 + i}",
            "status": "below_snr",
            "xmatch_class": "NEW_CANDIDATE",
            "period": 8.0,
            "depth": 0.001,
            "snr": 3.0,
            "is_valid": False,
        })
    return pd.DataFrame(rows)


class TestFindAlertable:
    """Test the alert filtering logic."""

    def test_filters_new_candidates_only(self) -> None:
        """Only NEW_CANDIDATE with valid + high SNR should pass."""
        df = _make_summary_df(n_new=2, n_known=1, n_low_snr=1)
        result = find_alertable_candidates(df)

        assert len(result) == 2
        assert all(result["xmatch_class"] == "NEW_CANDIDATE")

    def test_excludes_low_snr(self) -> None:
        """NEW_CANDIDATE with SNR < 7 must not trigger an alert."""
        df = _make_summary_df(n_new=0, n_known=0, n_low_snr=3)
        result = find_alertable_candidates(df)

        assert len(result) == 0

    def test_excludes_known_match(self) -> None:
        """KNOWN_MATCH candidates must not trigger alerts."""
        df = _make_summary_df(n_new=0, n_known=5, n_low_snr=0)
        result = find_alertable_candidates(df)

        assert len(result) == 0

    def test_empty_dataframe(self) -> None:
        """Empty input must return empty without crashing."""
        result = find_alertable_candidates(pd.DataFrame())
        assert result.empty


class TestAlertPayload:
    """Test the alert payload construction."""

    def test_payload_structure(self) -> None:
        """Payload must have required top-level keys."""
        df = _make_summary_df(n_new=1)
        alertable = find_alertable_candidates(df)
        payload = _build_alert_payload(alertable, sector=56)

        assert payload["alert_type"] == "new_candidates"
        assert payload["sector"] == 56
        assert payload["n_candidates"] == 1
        assert "timestamp" in payload
        assert "candidates" in payload

    def test_candidate_fields(self) -> None:
        """Each candidate in the payload must have core fields."""
        df = _make_summary_df(n_new=1)
        alertable = find_alertable_candidates(df)
        payload = _build_alert_payload(alertable)

        c = payload["candidates"][0]
        assert "tic_id" in c
        assert "period" in c
        assert "snr" in c

    def test_payload_is_json_serializable(self) -> None:
        """Payload must be JSON-serializable (for file + webhook)."""
        df = _make_summary_df(n_new=3)
        alertable = find_alertable_candidates(df)
        payload = _build_alert_payload(alertable, sector=1)

        serialized = json.dumps(payload)
        assert len(serialized) > 0


class TestSaveAlertFile:
    """Test file-based alert output."""

    def test_creates_json_file(self, tmp_path: Path) -> None:
        """save_alert_file must create a valid JSON file."""
        import exohunter.config as cfg
        original_dir = cfg.ALERTS_DIR
        cfg.ALERTS_DIR = tmp_path

        try:
            payload = {"alert_type": "test", "n_candidates": 1, "candidates": [{"tic_id": "X"}]}
            path = save_alert_file(payload, sector=99)

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["alert_type"] == "test"
        finally:
            cfg.ALERTS_DIR = original_dir

    def test_filename_contains_sector(self, tmp_path: Path) -> None:
        """Alert filename must include the sector number."""
        import exohunter.config as cfg
        original_dir = cfg.ALERTS_DIR
        cfg.ALERTS_DIR = tmp_path

        try:
            payload = {"alert_type": "test", "n_candidates": 0, "candidates": []}
            path = save_alert_file(payload, sector=42)

            assert "sector_42" in path.name
        finally:
            cfg.ALERTS_DIR = original_dir


class TestCheckAndDispatch:
    """Test the end-to-end alert dispatch."""

    def test_returns_zero_for_no_alerts(self) -> None:
        """No alert-worthy candidates → return 0."""
        df = _make_summary_df(n_new=0, n_known=2, n_low_snr=1)
        n = check_and_dispatch_alerts(df, sector=1)
        assert n == 0

    def test_returns_count_for_alerts(self, tmp_path: Path) -> None:
        """Alert-worthy candidates → return count and write file."""
        import exohunter.config as cfg
        original_dir = cfg.ALERTS_DIR
        cfg.ALERTS_DIR = tmp_path

        try:
            df = _make_summary_df(n_new=3, n_known=0, n_low_snr=0)
            n = check_and_dispatch_alerts(df, sector=56)

            assert n == 3
            # Should have written a JSON file
            files = list(tmp_path.glob("alert_*.json"))
            assert len(files) == 1
        finally:
            cfg.ALERTS_DIR = original_dir

    def test_empty_dataframe_no_crash(self) -> None:
        """Empty input must not crash."""
        n = check_and_dispatch_alerts(pd.DataFrame(), sector=1)
        assert n == 0
