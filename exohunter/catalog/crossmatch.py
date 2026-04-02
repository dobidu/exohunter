"""Cross-matching with known exoplanet catalogs.

Compares ExoHunter candidates against known exoplanets from
the NASA Exoplanet Archive and ExoFOP to identify:
    - Confirmed planets (true positives)
    - Known false positives
    - New candidates (not in any catalog)

TODO: Students can extend this to query the NASA Exoplanet Archive
      TAP service in real-time via astroquery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrossMatchResult:
    """Result of cross-matching a candidate against known catalogs.

    Attributes:
        candidate: The ExoHunter candidate.
        match_found: Whether a match was found in any catalog.
        catalog_name: Name of the matching catalog entry (if any).
        catalog_period: Period from the catalog (days).
        period_difference: Absolute difference between candidate and
            catalog period (days).
        status: One of ``"confirmed"``, ``"known_fp"``, or ``"new"``.
    """

    candidate: TransitCandidate
    match_found: bool = False
    catalog_name: str = ""
    catalog_period: float = 0.0
    period_difference: float = 0.0
    status: str = "new"


# A small built-in reference table of well-known TESS planets for
# quick offline testing.  In production, this would be replaced by
# a query to the NASA Exoplanet Archive.
# TODO: Replace with astroquery.nasa_exoplanet_archive for live queries
KNOWN_PLANETS: dict[str, list[dict]] = {
    "TIC 150428135": [
        {"name": "TOI-700 b", "period": 9.977,  "depth": 0.00058},
        {"name": "TOI-700 c", "period": 16.051, "depth": 0.00078},
        {"name": "TOI-700 d", "period": 37.426, "depth": 0.00081},
    ],
    "TIC 261136679": [
        {"name": "TOI-175 b (L 98-59 b)", "period": 2.2531, "depth": 0.00040},
        {"name": "TOI-175 c (L 98-59 c)", "period": 3.6906, "depth": 0.00069},
        {"name": "TOI-175 d (L 98-59 d)", "period": 7.4507, "depth": 0.00058},
    ],
}


def crossmatch_candidate(
    candidate: TransitCandidate,
    period_tolerance: float = 0.1,
) -> CrossMatchResult:
    """Cross-match a single candidate against the known planet table.

    Args:
        candidate: The candidate to look up.
        period_tolerance: Maximum allowed period difference (days)
            for a match.

    Returns:
        A ``CrossMatchResult`` indicating whether the candidate
        matches a known planet.
    """
    tic_id = candidate.tic_id.strip()

    # Normalize TIC ID format
    if not tic_id.upper().startswith("TIC"):
        tic_id = f"TIC {tic_id}"

    known = KNOWN_PLANETS.get(tic_id, [])

    for planet in known:
        diff = abs(candidate.period - planet["period"])
        if diff <= period_tolerance:
            result = CrossMatchResult(
                candidate=candidate,
                match_found=True,
                catalog_name=planet["name"],
                catalog_period=planet["period"],
                period_difference=diff,
                status="confirmed",
            )
            logger.info(
                "Cross-match: %s (P=%.4f d) matches %s (P=%.4f d, ΔP=%.4f d)",
                candidate.tic_id,
                candidate.period,
                planet["name"],
                planet["period"],
                diff,
            )
            return result

    # No match found — this is either a new candidate or noise
    logger.info(
        "Cross-match: %s (P=%.4f d) — no match in known catalogs",
        candidate.tic_id,
        candidate.period,
    )
    return CrossMatchResult(candidate=candidate, status="new")


def crossmatch_batch(
    candidates: list[TransitCandidate],
    period_tolerance: float = 0.1,
) -> list[CrossMatchResult]:
    """Cross-match a list of candidates against known catalogs.

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
