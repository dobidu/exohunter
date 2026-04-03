"""Dataset download and preparation for ML training.

Downloads the Kepler KOI cumulative table and ExoFOP-TESS TOI catalog
from NASA's public archives, then prepares labeled DataFrames for
training and validation.

Data sources are defined in ``data/datasets_sources.json`` so that
URLs can be updated without changing code.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def load_sources_config() -> dict:
    """Load dataset source URLs from the JSON config file.

    Returns:
        Dict mapping dataset names to their URL and output path.
    """
    path = config.DATASETS_SOURCES
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset sources config not found at {path}. "
            f"Ensure data/datasets_sources.json exists in the project root."
        )
    with open(path) as f:
        return json.load(f)


def download_dataset(name: str, force: bool = False) -> Path:
    """Download a single dataset by name from the sources config.

    Args:
        name: Dataset key in ``datasets_sources.json``
            (e.g. ``"kepler_koi"`` or ``"exofop_toi"``).
        force: If ``True``, re-download even if the file exists.

    Returns:
        Path to the downloaded CSV file.
    """
    sources = load_sources_config()
    if name not in sources:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(sources.keys())}"
        )

    entry = sources[name]
    url = entry["url"]
    output_path = config.PROJECT_ROOT / entry["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Dataset '%s' already exists at %s (use force=True to re-download)", name, output_path)
        return output_path

    logger.info("Downloading '%s' from %s ...", name, entry["source"])
    urllib.request.urlretrieve(url, output_path)

    # Verify the file is valid CSV
    df = pd.read_csv(output_path, comment="#", nrows=5)
    logger.info("Downloaded '%s': %s (%d columns)", name, output_path, len(df.columns))
    return output_path


def download_all(force: bool = False) -> dict[str, Path]:
    """Download all datasets defined in the sources config.

    Args:
        force: If ``True``, re-download even if files exist.

    Returns:
        Dict mapping dataset names to their local file paths.
    """
    sources = load_sources_config()
    paths = {}
    for name in sources:
        paths[name] = download_dataset(name, force=force)
    return paths


# ---------------------------------------------------------------------------
# Kepler KOI preparation
# ---------------------------------------------------------------------------

def prepare_kepler_koi(path: Path | None = None) -> pd.DataFrame:
    """Load and prepare the Kepler KOI dataset for training.

    Reads the cumulative KOI table, maps dispositions to three classes
    (``planet``, ``eclipsing_binary``, ``false_positive``), drops
    unlabeled rows, and returns a clean DataFrame with standardized
    column names.

    Label mapping:
        - ``CONFIRMED`` → ``planet``
        - ``FALSE POSITIVE`` with ``koi_fpflag_ss=1`` → ``eclipsing_binary``
        - ``FALSE POSITIVE`` with ``koi_fpflag_ss=0`` → ``false_positive``
        - ``CANDIDATE`` → excluded from training (no ground truth)

    Args:
        path: Path to the Kepler KOI CSV file. If ``None``, uses
            the default location ``data/datasets/kepler_koi.csv``.

    Returns:
        A DataFrame with columns: ``period``, ``depth``, ``duration``,
        ``snr``, ``impact_param``, ``stellar_teff``, ``stellar_logg``,
        ``stellar_radius``, ``duration_period_ratio``, ``depth_log``,
        ``label``.
    """
    if path is None:
        path = config.DATASETS_DIR / "kepler_koi.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Kepler KOI dataset not found at {path}. "
            f"Run: python scripts/download_training_data.py"
        )

    logger.info("Loading Kepler KOI dataset from %s", path)
    df = pd.read_csv(path, comment="#")

    # Rename columns to the common ExoHunter feature schema
    column_map = {
        "koi_period": "period",
        "koi_depth": "depth_ppm",
        "koi_duration": "duration_hours",
        "koi_model_snr": "snr",
        "koi_impact": "impact_param",
        "koi_steff": "stellar_teff",
        "koi_slogg": "stellar_logg",
        "koi_srad": "stellar_radius",
        "koi_disposition": "disposition",
        "koi_fpflag_ss": "fp_stellar_eclipse",
        "koi_fpflag_nt": "fp_not_transit",
        "koi_fpflag_co": "fp_centroid_offset",
        "koi_fpflag_ec": "fp_ephemeris_contam",
    }
    df = df.rename(columns=column_map)

    # Map dispositions to three-class labels
    def _map_label(row: pd.Series) -> str | None:
        disp = row.get("disposition", "")
        if disp == "CONFIRMED":
            return "planet"
        elif disp == "FALSE POSITIVE":
            # Use the stellar eclipse flag to separate eclipsing binaries
            if row.get("fp_stellar_eclipse", 0) == 1:
                return "eclipsing_binary"
            return "false_positive"
        return None  # CANDIDATE or unknown — exclude

    df["label"] = df.apply(_map_label, axis=1)

    # Drop unlabeled rows (CANDIDATE)
    n_before = len(df)
    df = df.dropna(subset=["label"])
    logger.info(
        "Label mapping: %d → %d rows (dropped %d CANDIDATE/unknown)",
        n_before, len(df), n_before - len(df),
    )

    # Convert units to ExoHunter conventions
    # depth: ppm → fractional (e.g. 1000 ppm → 0.001)
    df["depth"] = df["depth_ppm"] / 1e6
    # duration: hours → days
    df["duration"] = df["duration_hours"] / 24.0

    # Engineered features
    df["duration_period_ratio"] = df["duration"] / df["period"]
    df["depth_log"] = np.log10(df["depth"].clip(lower=1e-10))

    # Drop rows with missing critical features
    feature_cols = [
        "period", "depth", "duration", "snr", "impact_param",
        "stellar_teff", "stellar_logg", "stellar_radius",
        "duration_period_ratio", "depth_log",
    ]
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < n_before:
        logger.info("Dropped %d rows with missing features", n_before - len(df))

    # Select final columns
    output_cols = feature_cols + ["label"]
    df = df[output_cols].reset_index(drop=True)

    # Log class distribution
    for label, count in df["label"].value_counts().items():
        logger.info("  %s: %d", label, count)

    return df


def prepare_exofop_toi(path: Path | None = None) -> pd.DataFrame:
    """Load and prepare the ExoFOP-TESS TOI dataset for validation.

    Maps TFOPWG dispositions to the same three classes used for
    training.  Only entries with definitive dispositions (KP, CP, FP)
    are included.

    Label mapping:
        - ``KP`` (Known Planet) + ``CP`` (Confirmed Planet) → ``planet``
        - ``FP`` (False Positive) → ``false_positive``
        - ``PC``, ``APC``, ``FA`` → excluded (uncertain)

    Args:
        path: Path to the ExoFOP TOI CSV. If ``None``, uses
            ``data/datasets/exofop_toi.csv``.

    Returns:
        A DataFrame with the same columns as ``prepare_kepler_koi``.
    """
    if path is None:
        path = config.DATASETS_DIR / "exofop_toi.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"ExoFOP TOI dataset not found at {path}. "
            f"Run: python scripts/download_training_data.py"
        )

    logger.info("Loading ExoFOP TOI dataset from %s", path)
    df = pd.read_csv(path, comment="#")

    # Normalize column names (ExoFOP format varies)
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower == "period (days)" or lower == "period":
            col_map[col] = "period"
        elif lower == "depth (ppm)" or lower == "depth ppm":
            col_map[col] = "depth_ppm"
        elif lower == "duration (hours)" or lower == "duration":
            col_map[col] = "duration_hours"
        elif "disposition" in lower:
            col_map[col] = "disposition"
    df = df.rename(columns=col_map)

    # Map dispositions
    disp_map = {"KP": "planet", "CP": "planet", "FP": "false_positive"}
    df["label"] = df["disposition"].map(disp_map)
    df = df.dropna(subset=["label"])

    # Convert units
    if "depth_ppm" in df.columns:
        df["depth_ppm"] = pd.to_numeric(df["depth_ppm"], errors="coerce")
        df["depth"] = df["depth_ppm"] / 1e6
    if "duration_hours" in df.columns:
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
        df["duration"] = df["duration_hours"] / 24.0
    if "period" in df.columns:
        df["period"] = pd.to_numeric(df["period"], errors="coerce")

    # Engineered features (some may be NaN — ExoFOP has fewer columns)
    df["duration_period_ratio"] = df["duration"] / df["period"]
    df["depth_log"] = np.log10(df["depth"].clip(lower=1e-10))

    # ExoFOP doesn't have SNR, impact, or stellar params in all rows.
    # Fill missing with NaN — the model handles this via imputation.
    for col in ["snr", "impact_param", "stellar_teff", "stellar_logg", "stellar_radius"]:
        if col not in df.columns:
            df[col] = np.nan

    feature_cols = [
        "period", "depth", "duration", "snr", "impact_param",
        "stellar_teff", "stellar_logg", "stellar_radius",
        "duration_period_ratio", "depth_log",
    ]
    output_cols = feature_cols + ["label"]

    # Drop rows with missing period/depth/duration (critical)
    df = df.dropna(subset=["period", "depth", "duration"])
    df = df[output_cols].reset_index(drop=True)

    for label, count in df["label"].value_counts().items():
        logger.info("  %s: %d", label, count)

    return df
