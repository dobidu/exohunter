"""Cross-matching with the ExoFOP-TESS TOI catalog.

Compares ExoHunter candidates against the list of known TESS Objects
of Interest (TOIs) from the ExoFOP-TESS database to classify each
candidate into one of four categories:

    - **KNOWN_MATCH**: TIC ID is in the TOI catalog AND the detected
      period matches a known TOI period within tolerance.  This is a
      re-detection of a known signal — good for pipeline validation.

    - **KNOWN_TOI**: TIC ID is in the TOI catalog but our detected
      period does not match any catalogued period.  Could be a new
      planet in a known multi-planet system, or a systematic artefact.

    - **HARMONIC**: Our detected period is a harmonic (2×, 3×, 0.5×,
      1/3×) of a known TOI period.  Likely an alias, not a new planet.

    - **NEW_CANDIDATE**: TIC ID is NOT in any catalog.  This is the
      most exciting outcome — a potentially undiscovered exoplanet.

The TOI catalog is loaded from a static CSV downloaded from ExoFOP:
    https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv

To update the catalog, run::

    python -m exohunter.catalog.crossmatch --update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from exohunter import config
from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification enum
# ---------------------------------------------------------------------------

class MatchClass(str, Enum):
    """Cross-match classification for a transit candidate."""

    NEW_CANDIDATE = "NEW_CANDIDATE"
    """TIC ID not in any catalog — potential new exoplanet."""

    KNOWN_MATCH = "KNOWN_MATCH"
    """TIC ID and period match a known TOI — re-detection."""

    KNOWN_TOI = "KNOWN_TOI"
    """TIC ID is a known TOI but period differs — different signal."""

    HARMONIC = "HARMONIC"
    """Period is a harmonic of a known TOI period — likely alias."""


# ---------------------------------------------------------------------------
# Cross-match result
# ---------------------------------------------------------------------------

@dataclass
class CrossMatchResult:
    """Result of cross-matching a candidate against the TOI catalog.

    Attributes:
        candidate: The ExoHunter candidate.
        match_class: Classification (see ``MatchClass``).
        match_found: Whether a match was found in the catalog.
        catalog_name: Name of the matching catalog entry (if any).
        catalog_period: Period from the catalog (days).
        period_difference: Absolute difference in period (days).
        status: Legacy string field — kept for backward compatibility.
    """

    candidate: TransitCandidate
    match_class: MatchClass = MatchClass.NEW_CANDIDATE
    match_found: bool = False
    catalog_name: str = ""
    catalog_period: float = 0.0
    period_difference: float = 0.0
    status: str = "new"

    def __post_init__(self) -> None:
        """Sync the legacy ``status`` field with ``match_class``."""
        self.status = self.match_class.value


# ---------------------------------------------------------------------------
# TOI catalog loader
# ---------------------------------------------------------------------------

_TOI_CATALOG_PATH: Path = config.CATALOG_DIR / "toi_catalog.csv"

# In-memory cache of the loaded catalog (loaded once per process)
_toi_cache: pd.DataFrame | None = None


def _download_toi_catalog(output_path: Path | None = None) -> Path:
    """Download the TOI catalog from ExoFOP-TESS.

    Args:
        output_path: Where to save the CSV. Defaults to
            ``data/catalogs/toi_catalog.csv``.

    Returns:
        Path to the downloaded file.
    """
    import urllib.request

    if output_path is None:
        output_path = _TOI_CATALOG_PATH

    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    logger.info("Downloading TOI catalog from ExoFOP-TESS...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, output_path)
        # Verify the file is valid CSV
        df = pd.read_csv(output_path, comment="#")
        logger.info(
            "TOI catalog downloaded: %d entries, saved to %s",
            len(df), output_path,
        )
        return output_path
    except Exception:
        logger.exception("Failed to download TOI catalog")
        raise


def load_toi_catalog(force_reload: bool = False) -> pd.DataFrame:
    """Load the TOI catalog into memory.

    Reads the static CSV from ``data/catalogs/toi_catalog.csv``.
    The catalog is cached in memory after the first load.

    The relevant columns are:
        - ``TIC ID``: numeric TIC identifier
        - ``Period (days)``: orbital period
        - ``TOI``: TOI number (e.g. 700.01)
        - ``Depth (ppm)``: transit depth in parts per million
        - ``TFOPWG Disposition``: community disposition (PC, FP, KP, etc.)

    Args:
        force_reload: If ``True``, re-read from disk even if cached.

    Returns:
        A DataFrame with one row per TOI entry.
    """
    global _toi_cache

    if _toi_cache is not None and not force_reload:
        return _toi_cache

    if not _TOI_CATALOG_PATH.exists():
        logger.warning(
            "TOI catalog not found at %s — cross-matching will use "
            "built-in reference table only. Run "
            "'python -m exohunter.catalog.crossmatch --update' to download.",
            _TOI_CATALOG_PATH,
        )
        _toi_cache = pd.DataFrame()
        return _toi_cache

    try:
        df = pd.read_csv(_TOI_CATALOG_PATH, comment="#")

        # Normalize column names (ExoFOP uses varying formats)
        col_map = {}
        for col in df.columns:
            lower = col.strip().lower()
            if lower == "tic id" or lower == "tic_id":
                col_map[col] = "tic_id"
            elif lower == "period (days)" or lower == "period":
                col_map[col] = "period"
            elif lower == "toi":
                col_map[col] = "toi"
            elif lower == "depth (ppm)" or lower == "depth ppm":
                col_map[col] = "depth_ppm"
            elif "disposition" in lower:
                col_map[col] = "disposition"

        df = df.rename(columns=col_map)

        # Ensure TIC IDs are integers for fast lookup
        if "tic_id" in df.columns:
            df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
            df = df.dropna(subset=["tic_id"])
            df["tic_id"] = df["tic_id"].astype(int)

        # Convert period to float, dropping entries without a period
        if "period" in df.columns:
            df["period"] = pd.to_numeric(df["period"], errors="coerce")

        _toi_cache = df
        logger.info("Loaded TOI catalog: %d entries from %s", len(df), _TOI_CATALOG_PATH)
        return _toi_cache

    except Exception:
        logger.exception("Failed to parse TOI catalog at %s", _TOI_CATALOG_PATH)
        _toi_cache = pd.DataFrame()
        return _toi_cache


def _extract_tic_number(tic_id: str) -> int | None:
    """Extract the numeric part of a TIC ID string.

    Args:
        tic_id: e.g. ``"TIC 150428135"`` or ``"150428135"``.

    Returns:
        The integer TIC number, or ``None`` if parsing fails.
    """
    try:
        cleaned = tic_id.upper().replace("TIC", "").strip()
        return int(cleaned)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Built-in reference table (offline fallback)
# ---------------------------------------------------------------------------

KNOWN_PLANETS: dict[int, list[dict]] = {
    150428135: [
        {"name": "TOI-700 b",  "period": 9.977,  "depth_ppm": 580},
        {"name": "TOI-700 c",  "period": 16.051, "depth_ppm": 780},
        {"name": "TOI-700 d",  "period": 37.426, "depth_ppm": 810},
    ],
    261136679: [
        {"name": "TOI-175 b (L 98-59 b)", "period": 2.2531, "depth_ppm": 400},
        {"name": "TOI-175 c (L 98-59 c)", "period": 3.6906, "depth_ppm": 690},
        {"name": "TOI-175 d (L 98-59 d)", "period": 7.4507, "depth_ppm": 580},
    ],
}


# ---------------------------------------------------------------------------
# Cross-matching logic
# ---------------------------------------------------------------------------

# Harmonic ratios to test: if candidate_period / known_period is close
# to any of these, the candidate is classified as HARMONIC.
_HARMONIC_RATIOS: list[float] = [0.5, 2.0, 1.0 / 3.0, 3.0]


def crossmatch_candidate(
    candidate: TransitCandidate,
    period_tolerance: float = 0.1,
    harmonic_tolerance: float = 0.02,
) -> CrossMatchResult:
    """Cross-match a single candidate against the TOI catalog.

    Classification logic:
        1. Extract numeric TIC ID from the candidate.
        2. Look up the TIC ID in the loaded TOI catalog.
        3. If found, compare periods:
           a. Period matches within tolerance → ``KNOWN_MATCH``
           b. Period is a harmonic → ``HARMONIC``
           c. Neither → ``KNOWN_TOI`` (different signal in known system)
        4. If not found in TOI catalog, check the built-in reference
           table (for offline testing).
        5. If not found anywhere → ``NEW_CANDIDATE``

    Args:
        candidate: The candidate to classify.
        period_tolerance: Maximum period difference (days) for a
            ``KNOWN_MATCH`` classification.
        harmonic_tolerance: Relative tolerance for harmonic matching.
            A ratio within ``harmonic_tolerance`` of a harmonic integer
            ratio (2:1, 3:1, 1:2, 1:3) triggers a ``HARMONIC`` flag.

    Returns:
        A ``CrossMatchResult`` with the classification.
    """
    tic_num = _extract_tic_number(candidate.tic_id)
    if tic_num is None:
        logger.warning("Could not parse TIC ID: %s", candidate.tic_id)
        return CrossMatchResult(
            candidate=candidate,
            match_class=MatchClass.NEW_CANDIDATE,
        )

    # Try the TOI catalog first (if loaded)
    toi_df = load_toi_catalog()
    known_periods: list[dict] = []

    if not toi_df.empty and "tic_id" in toi_df.columns:
        matches = toi_df[toi_df["tic_id"] == tic_num]
        if not matches.empty:
            for _, row in matches.iterrows():
                entry: dict = {"name": f"TOI-{row.get('toi', '?')}"}
                period_val = row.get("period")
                if period_val is not None and not np.isnan(period_val):
                    entry["period"] = float(period_val)
                known_periods.append(entry)

    # Fall back to built-in reference table
    if not known_periods and tic_num in KNOWN_PLANETS:
        known_periods = KNOWN_PLANETS[tic_num]

    # If TIC is not in any catalog → NEW_CANDIDATE
    if not known_periods:
        logger.info(
            "Cross-match: %s (P=%.4f d) → NEW_CANDIDATE (not in any catalog)",
            candidate.tic_id, candidate.period,
        )
        return CrossMatchResult(
            candidate=candidate,
            match_class=MatchClass.NEW_CANDIDATE,
        )

    # TIC is in the catalog — check if the period matches
    for entry in known_periods:
        cat_period = entry.get("period")
        if cat_period is None:
            continue

        diff = abs(candidate.period - cat_period)

        # Direct period match
        if diff <= period_tolerance:
            result = CrossMatchResult(
                candidate=candidate,
                match_class=MatchClass.KNOWN_MATCH,
                match_found=True,
                catalog_name=entry.get("name", ""),
                catalog_period=cat_period,
                period_difference=diff,
            )
            logger.info(
                "Cross-match: %s (P=%.4f d) → KNOWN_MATCH with %s "
                "(P=%.4f d, ΔP=%.4f d)",
                candidate.tic_id, candidate.period,
                entry.get("name", "?"), cat_period, diff,
            )
            return result

        # Harmonic check
        ratio = candidate.period / cat_period
        for h in _HARMONIC_RATIOS:
            if abs(ratio - h) / h < harmonic_tolerance:
                harmonic_label = f"{int(1/h) if h < 1 else int(h)}:1"
                result = CrossMatchResult(
                    candidate=candidate,
                    match_class=MatchClass.HARMONIC,
                    match_found=True,
                    catalog_name=entry.get("name", ""),
                    catalog_period=cat_period,
                    period_difference=diff,
                )
                logger.info(
                    "Cross-match: %s (P=%.4f d) → HARMONIC (%s) of %s "
                    "(P=%.4f d)",
                    candidate.tic_id, candidate.period,
                    harmonic_label, entry.get("name", "?"), cat_period,
                )
                return result

    # TIC is in catalog but no period match → KNOWN_TOI
    logger.info(
        "Cross-match: %s (P=%.4f d) → KNOWN_TOI (TIC in catalog, "
        "period does not match any of %d known TOIs)",
        candidate.tic_id, candidate.period, len(known_periods),
    )
    return CrossMatchResult(
        candidate=candidate,
        match_class=MatchClass.KNOWN_TOI,
        match_found=True,
        catalog_name=known_periods[0].get("name", ""),
    )


def crossmatch_batch(
    candidates: list[TransitCandidate],
    period_tolerance: float = 0.1,
) -> list[CrossMatchResult]:
    """Cross-match a list of candidates against the TOI catalog.

    Args:
        candidates: List of candidates to check.
        period_tolerance: Period matching tolerance in days.

    Returns:
        List of ``CrossMatchResult`` objects.
    """
    return [
        crossmatch_candidate(c, period_tolerance=period_tolerance)
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# CLI: download/update the TOI catalog
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ExoHunter — TOI catalog management",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Download the latest TOI catalog from ExoFOP-TESS",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show info about the currently loaded catalog",
    )
    args = parser.parse_args()

    if args.update:
        path = _download_toi_catalog()
        print(f"TOI catalog saved to: {path}")

    if args.info or not args.update:
        df = load_toi_catalog(force_reload=True)
        if df.empty:
            print("No TOI catalog loaded. Run with --update to download.")
        else:
            print(f"TOI catalog: {len(df)} entries")
            if "tic_id" in df.columns:
                print(f"Unique TICs: {df['tic_id'].nunique()}")
            if "disposition" in df.columns:
                print(f"\nDisposition breakdown:")
                for disp, count in df["disposition"].value_counts().items():
                    print(f"  {disp}: {count}")
