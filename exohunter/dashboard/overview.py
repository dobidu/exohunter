"""Data overview utilities for the dashboard.

Scans the data directories and builds summary information for
the Data Overview section: cache statistics, batch results index,
reports gallery, alerts feed, and ML model status.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def scan_cache_stats() -> dict:
    """Scan the light curve cache directory for statistics.

    Returns:
        Dict with ``n_files``, ``total_size_mb``, ``largest_files``
        (top 5 by size with name and MB).
    """
    cache_dir = config.CACHE_DIR
    if not cache_dir.exists():
        return {"n_files": 0, "total_size_mb": 0, "largest_files": []}

    files = list(cache_dir.glob("*.fits"))
    total_bytes = sum(f.stat().st_size for f in files)

    largest = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:5]
    largest_info = [
        {"name": f.stem.replace("TIC_", "TIC "), "size_mb": f.stat().st_size / 1e6}
        for f in largest
    ]

    return {
        "n_files": len(files),
        "total_size_mb": round(total_bytes / 1e6, 1),
        "largest_files": largest_info,
    }


def scan_batch_results() -> list[dict]:
    """Scan data/results/ for batch CSV files with metadata.

    Returns:
        List of dicts, each with ``filename``, ``sector``, ``date``,
        ``n_targets``, ``n_validated``, ``n_new_candidate``, ``mode``.
    """
    results_dir = config.RESULTS_DIR
    if not results_dir.exists():
        return []

    results = []
    for csv_path in sorted(results_dir.glob("sector_*.csv")):
        if "_candidates" in csv_path.stem:
            continue  # skip the _candidates subset files

        try:
            df = pd.read_csv(csv_path, nrows=None)
            mtime = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc)

            # Extract sector number from filename
            parts = csv_path.stem.split("_")
            sector = parts[1] if len(parts) >= 2 else "?"
            mode = "multi-sector" if "multi" in csv_path.stem else "standard"

            n_validated = 0
            n_new = 0
            if "xmatch_class" in df.columns:
                n_new = int((df["xmatch_class"] == "NEW_CANDIDATE").sum())
            if "is_valid" in df.columns:
                n_validated = int(df["is_valid"].sum())

            results.append({
                "filename": csv_path.name,
                "sector": sector,
                "date": mtime.strftime("%Y-%m-%d %H:%M"),
                "n_targets": len(df),
                "n_validated": n_validated,
                "n_new_candidate": n_new,
                "mode": mode,
            })
        except Exception:
            logger.warning("Could not parse %s", csv_path)

    return results


def scan_reports() -> list[dict]:
    """Scan data/reports/ for PNG diagnostic reports.

    Returns:
        List of dicts with ``filename``, ``tic_id``, ``path``, ``size_kb``.
    """
    reports_dir = config.REPORTS_DIR
    if not reports_dir.exists():
        return []

    reports = []
    for png_path in sorted(reports_dir.glob("*.png")):
        tic_id = png_path.stem.replace("TIC_", "TIC ")
        reports.append({
            "filename": png_path.name,
            "tic_id": tic_id,
            "path": str(png_path),
            "size_kb": round(png_path.stat().st_size / 1024, 1),
        })

    return reports


def load_report_as_base64(path: str) -> str | None:
    """Load a PNG report and encode it as base64 for inline display.

    Args:
        path: Path to the PNG file.

    Returns:
        A data URI string for use in an ``html.Img`` src attribute,
        or ``None`` if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return None

    with open(p, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def scan_alerts() -> list[dict]:
    """Scan data/alerts/ for alert JSON files.

    Returns:
        List of dicts with ``filename``, ``timestamp``, ``sector``,
        ``n_candidates``, ``candidates`` (list of TIC IDs).
    """
    alerts_dir = config.ALERTS_DIR
    if not alerts_dir.exists():
        return []

    alerts = []
    for json_path in sorted(alerts_dir.glob("alert_*.json"), reverse=True):
        try:
            with open(json_path) as f:
                data = json.load(f)

            tic_ids = [c.get("tic_id", "?") for c in data.get("candidates", [])]

            alerts.append({
                "filename": json_path.name,
                "timestamp": data.get("timestamp", ""),
                "sector": data.get("sector"),
                "n_candidates": data.get("n_candidates", 0),
                "candidates": tic_ids[:5],
            })
        except Exception:
            logger.warning("Could not parse alert %s", json_path)

    return alerts


def scan_ml_status() -> dict:
    """Check which ML models are trained and available.

    Returns:
        Dict with ``rf_available``, ``cnn_available``, ``rf_path``,
        ``cnn_path``, ``rf_size_kb``, ``cnn_size_kb``.
    """
    rf_path = config.MODELS_DIR / "transit_classifier.joblib"
    cnn_path = config.MODELS_DIR / "transit_cnn.pt"

    return {
        "rf_available": rf_path.exists(),
        "rf_path": str(rf_path),
        "rf_size_kb": round(rf_path.stat().st_size / 1024, 1) if rf_path.exists() else 0,
        "cnn_available": cnn_path.exists(),
        "cnn_path": str(cnn_path),
        "cnn_size_kb": round(cnn_path.stat().st_size / 1024, 1) if cnn_path.exists() else 0,
    }
