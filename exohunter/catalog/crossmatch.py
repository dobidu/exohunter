"""Cross-matching with the TESS TOI catalog (live + offline).

Compares ExoHunter candidates against known TESS Objects of Interest
(TOIs) to classify each candidate into one of four categories:

    - **KNOWN_MATCH**: TIC ID + period match a known TOI.
    - **KNOWN_TOI**: TIC ID is in the catalog but at a different period.
    - **HARMONIC**: Period is a harmonic (2×, 3×, 0.5×, 1/3×) of a known TOI.
    - **NEW_CANDIDATE**: TIC ID is NOT in any catalog.

Catalog loading uses a three-tier fallback strategy:

    1. **NASA Exoplanet Archive TAP API** — live query for the latest
       TOI table.  The result is cached to disk and re-used until it
       exceeds ``TOI_CATALOG_MAX_AGE_HOURS`` (default 48 h).
    2. **Local CSV** — a previously downloaded static CSV from ExoFOP
       or the TAP cache file (``data/catalogs/toi_catalog.csv``).
    3. **Built-in reference table** — a small hardcoded table of
       well-known planets for fully offline testing.

To force a refresh::

    python -m exohunter.catalog.crossmatch --update
"""

from __future__ import annotations

import time as time_module
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
# TOI catalog loading — three-tier fallback
# ---------------------------------------------------------------------------

_TOI_CATALOG_PATH: Path = config.CATALOG_DIR / "toi_catalog.csv"

# In-memory cache (loaded once per process unless force_reload)
_toi_cache: pd.DataFrame | None = None


def _is_catalog_stale(path: Path, max_age_hours: float) -> bool:
    """Check if a cached catalog file is older than the allowed age.

    Args:
        path: Path to the cached CSV.
        max_age_hours: Maximum age in hours before the file is stale.

    Returns:
        ``True`` if the file does not exist or is older than
        ``max_age_hours``.
    """
    if not path.exists():
        return True

    if max_age_hours <= 0:
        return True  # 0 means always refresh

    age_seconds = time_module.time() - path.stat().st_mtime
    age_hours = age_seconds / 3600.0
    return age_hours > max_age_hours


def _fetch_toi_via_tap(output_path: Path | None = None) -> pd.DataFrame | None:
    """Fetch the TOI catalog from the NASA Exoplanet Archive via TAP.

    Uses the ``astroquery.ipac.nexsci.nasa_exoplanet_archive`` TAP
    service to query the ``toi`` table for all TOIs with their TIC IDs,
    periods, and dispositions.

    Args:
        output_path: Where to cache the result as CSV. If ``None``,
            uses ``data/catalogs/toi_catalog.csv``.

    Returns:
        A pandas DataFrame with the TOI data, or ``None`` on failure.
    """
    if output_path is None:
        output_path = _TOI_CATALOG_PATH

    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        logger.info("Querying NASA Exoplanet Archive (TAP) for TOI catalog...")

        table = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid,toi,pl_orbper,pl_trandep,pl_trandur,tfopwg_disp",
        )

        if table is None or len(table) == 0:
            logger.warning("TAP query returned no results")
            return None

        df = table.to_pandas()

        # Rename to the ExoHunter standard column names
        col_map = {
            "tid": "tic_id",
            "toi": "toi",
            "pl_orbper": "period",
            "pl_trandep": "depth_ppm",
            "pl_trandur": "duration_hours",
            "tfopwg_disp": "disposition",
        }
        df = df.rename(columns=col_map)

        # Ensure TIC IDs are integers
        if "tic_id" in df.columns:
            df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
            df = df.dropna(subset=["tic_id"])
            df["tic_id"] = df["tic_id"].astype(int)

        if "period" in df.columns:
            df["period"] = pd.to_numeric(df["period"], errors="coerce")

        # Cache to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(
            "TOI catalog fetched via TAP: %d entries, cached to %s",
            len(df), output_path,
        )
        return df

    except ImportError:
        logger.warning(
            "astroquery.ipac.nexsci not available — cannot query TAP. "
            "Install astroquery >= 0.4 for live queries."
        )
        return None
    except Exception:
        logger.warning("TAP query failed — will try local CSV fallback", exc_info=True)
        return None


def _download_toi_exofop(output_path: Path | None = None) -> Path:
    """Download the TOI catalog from ExoFOP-TESS (HTTP fallback).

    This is the legacy download method, kept as a second fallback
    if the TAP query is unavailable.

    Args:
        output_path: Where to save the CSV.

    Returns:
        Path to the downloaded file.
    """
    import urllib.request

    if output_path is None:
        output_path = _TOI_CATALOG_PATH

    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    logger.info("Downloading TOI catalog from ExoFOP-TESS (HTTP)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)

    df = pd.read_csv(output_path, comment="#")
    logger.info("TOI catalog downloaded (ExoFOP): %d entries", len(df))
    return output_path


def _parse_local_csv(path: Path) -> pd.DataFrame:
    """Parse a local TOI catalog CSV into a normalized DataFrame.

    Handles both TAP-format (``tic_id``, ``period``) and ExoFOP-format
    (``TIC ID``, ``Period (days)``) column names.

    Args:
        path: Path to the CSV file.

    Returns:
        A normalized DataFrame with ``tic_id`` (int), ``period`` (float),
        ``toi``, ``disposition`` columns.
    """
    df = pd.read_csv(path, comment="#")

    # Normalize column names from either format
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("tic id", "tic_id", "tid"):
            col_map[col] = "tic_id"
        elif lower in ("period (days)", "period", "pl_orbper"):
            col_map[col] = "period"
        elif lower == "toi":
            col_map[col] = "toi"
        elif lower in ("depth (ppm)", "depth ppm", "pl_trandep"):
            col_map[col] = "depth_ppm"
        elif "disposition" in lower or lower == "tfopwg_disp":
            col_map[col] = "disposition"
    df = df.rename(columns=col_map)

    if "tic_id" in df.columns:
        df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
        df = df.dropna(subset=["tic_id"])
        df["tic_id"] = df["tic_id"].astype(int)

    if "period" in df.columns:
        df["period"] = pd.to_numeric(df["period"], errors="coerce")

    return df


def load_toi_catalog(
    force_reload: bool = False,
    source: str = "auto",
) -> pd.DataFrame:
    """Load the TOI catalog with three-tier fallback.

    Loading strategy (``source="auto"``):
        1. If the cached CSV exists and is fresh (< ``TOI_CATALOG_MAX_AGE_HOURS``),
           load it from disk.
        2. If stale or missing, try a live TAP query to the NASA
           Exoplanet Archive (caches the result to disk).
        3. If TAP fails, try downloading from ExoFOP-TESS (HTTP).
        4. If that also fails, load whatever CSV exists on disk
           (even if stale).
        5. If no CSV exists at all, return an empty DataFrame
           (the built-in reference table will be used as final fallback
           in ``crossmatch_candidate``).

    Args:
        force_reload: If ``True``, bypass the in-memory cache and
            re-read from disk (or re-fetch from network).
        source: Loading strategy:
            - ``"auto"`` — three-tier fallback (default)
            - ``"tap"`` — force TAP query
            - ``"csv"`` — load local CSV only (no network)

    Returns:
        A DataFrame with one row per TOI entry.
    """
    global _toi_cache

    if _toi_cache is not None and not force_reload:
        return _toi_cache

    max_age = config.TOI_CATALOG_MAX_AGE_HOURS
    df = pd.DataFrame()

    if source == "csv":
        # Local CSV only — no network
        if _TOI_CATALOG_PATH.exists():
            df = _parse_local_csv(_TOI_CATALOG_PATH)
            logger.info("Loaded TOI catalog (local CSV): %d entries", len(df))
        else:
            logger.warning(
                "TOI catalog not found at %s — using built-in reference only",
                _TOI_CATALOG_PATH,
            )

    elif source == "tap":
        # Force TAP query
        tap_df = _fetch_toi_via_tap()
        if tap_df is not None:
            df = tap_df
        else:
            logger.warning("TAP query failed — no catalog loaded")

    else:
        # Auto: check freshness → TAP → ExoFOP HTTP → stale CSV
        if not _is_catalog_stale(_TOI_CATALOG_PATH, max_age):
            df = _parse_local_csv(_TOI_CATALOG_PATH)
            logger.info(
                "Loaded TOI catalog (cached, <%.0fh old): %d entries",
                max_age, len(df),
            )
        else:
            # Try TAP first
            tap_df = _fetch_toi_via_tap()
            if tap_df is not None:
                df = tap_df
            else:
                # Try ExoFOP HTTP download
                try:
                    _download_toi_exofop()
                    df = _parse_local_csv(_TOI_CATALOG_PATH)
                except Exception:
                    logger.warning("ExoFOP download also failed", exc_info=True)
                    # Last resort: load stale CSV if it exists
                    if _TOI_CATALOG_PATH.exists():
                        df = _parse_local_csv(_TOI_CATALOG_PATH)
                        logger.info(
                            "Using stale TOI catalog from disk: %d entries", len(df),
                        )

    _toi_cache = df
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
# CLI: update/inspect the TOI catalog
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ExoHunter — TOI catalog management",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch the latest TOI catalog (TAP first, then ExoFOP HTTP)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "tap", "csv"],
        default="auto",
        help="Catalog source: auto (TAP → CSV), tap (TAP only), csv (local only)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show info about the currently loaded catalog",
    )
    args = parser.parse_args()

    if args.update:
        df = load_toi_catalog(force_reload=True, source=args.source)
        if not df.empty:
            print(f"TOI catalog loaded: {len(df)} entries")
        else:
            print("Failed to load TOI catalog.")

    if args.info or not args.update:
        df = load_toi_catalog(force_reload=True, source=args.source)
        if df.empty:
            print("No TOI catalog loaded. Run with --update to fetch.")
        else:
            print(f"TOI catalog: {len(df)} entries")
            if "tic_id" in df.columns:
                print(f"Unique TICs: {df['tic_id'].nunique()}")
            if "disposition" in df.columns:
                print(f"\nDisposition breakdown:")
                for disp, count in df["disposition"].value_counts().items():
                    print(f"  {disp}: {count}")
            # Show freshness
            if _TOI_CATALOG_PATH.exists():
                age_hours = (time_module.time() - _TOI_CATALOG_PATH.stat().st_mtime) / 3600
                max_age = config.TOI_CATALOG_MAX_AGE_HOURS
                status = "fresh" if age_hours <= max_age else "stale"
                print(f"\nCache age: {age_hours:.1f} hours ({status}, max={max_age}h)")
