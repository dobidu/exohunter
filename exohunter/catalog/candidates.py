"""Candidate catalog management.

Stores, filters, and retrieves transit candidates discovered by the
pipeline.  The catalog is an in-memory list backed by optional
persistence to CSV.

Includes a scoring system that ranks candidates by how promising they
are for visual inspection and follow-up observations.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from exohunter import config
from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult, validate_candidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def compute_score(
    candidate: TransitCandidate,
    validation: ValidationResult,
) -> float:
    """Compute a priority score for a transit candidate.

    The score combines SNR (signal strength) with penalty factors
    for characteristics that suggest false positives:

        score = SNR × v_shape_factor × depth_factor

    Factors:
        - **V-shape**: 1.0 if the transit is box-like (v_shape test
          passed, metric < 0.5), 0.5 if V-shaped (suggests eclipsing
          binary rather than a planet).
        - **Depth**: 1.0 if depth < 2% (plausible for a planet),
          0.7 if depth >= 2% (deep transits are more likely to be
          eclipsing binaries or blended systems).

    Higher scores are more promising for follow-up.

    Args:
        candidate: The transit candidate.
        validation: The validation result (provides V-shape test).

    Returns:
        A non-negative score.  Typical range: 3.5–50+ for validated
        candidates.
    """
    # V-shape factor: penalise V-shaped dips (possible eclipsing binary)
    v_shape_passed = validation.tests.get("v_shape", True)
    v_shape_factor = 1.0 if v_shape_passed else 0.5

    # Depth factor: penalise very deep transits (> 2%)
    depth_factor = 1.0 if candidate.depth < 0.02 else 0.7

    score = candidate.snr * v_shape_factor * depth_factor
    return round(score, 2)


class CandidateCatalog:
    """In-memory catalog of transit candidates with validation status.

    Attributes:
        candidates: List of ``(TransitCandidate, ValidationResult)`` tuples.
    """

    def __init__(self) -> None:
        self.candidates: list[tuple[TransitCandidate, ValidationResult]] = []

    def add(
        self,
        candidate: TransitCandidate,
        validation: ValidationResult | None = None,
    ) -> None:
        """Add a candidate to the catalog.

        If no ``ValidationResult`` is provided, a default validation
        is run using ``validate_candidate`` with no light curve data
        (V-shape and harmonic checks are skipped).

        Args:
            candidate: The transit candidate to add.
            validation: Pre-computed validation result, if available.
        """
        if validation is None:
            existing = [c for c, _ in self.candidates]
            validation = validate_candidate(
                candidate, other_candidates=existing
            )

        self.candidates.append((candidate, validation))
        score = compute_score(candidate, validation)
        logger.info(
            "Added candidate %s (P=%.4f d, valid=%s, score=%.1f) — catalog size: %d",
            candidate.tic_id,
            candidate.period,
            validation.is_valid,
            score,
            len(self.candidates),
        )

    def get_valid(self) -> list[TransitCandidate]:
        """Return only validated candidates.

        Returns:
            List of ``TransitCandidate`` objects that passed validation.
        """
        return [c for c, v in self.candidates if v.is_valid]

    def get_rejected(self) -> list[TransitCandidate]:
        """Return only rejected candidates.

        Returns:
            List of ``TransitCandidate`` objects that failed validation.
        """
        return [c for c, v in self.candidates if not v.is_valid]

    def get_ranked(self, limit: int | None = None) -> list[tuple[TransitCandidate, ValidationResult, float]]:
        """Return candidates sorted by score (highest first).

        Args:
            limit: Maximum number of candidates to return.
                ``None`` returns all.

        Returns:
            List of ``(candidate, validation, score)`` tuples,
            sorted by descending score.
        """
        scored = [
            (c, v, compute_score(c, v))
            for c, v in self.candidates
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        if limit is not None:
            scored = scored[:limit]

        return scored

    def get_top(self, n: int = 20) -> list[tuple[TransitCandidate, ValidationResult, float]]:
        """Return the top N most promising candidates for visual inspection.

        Convenience wrapper around ``get_ranked`` that defaults to 20.

        Args:
            n: Number of top candidates to return.

        Returns:
            List of ``(candidate, validation, score)`` tuples.
        """
        return self.get_ranked(limit=n)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the catalog to a pandas DataFrame.

        Returns:
            A DataFrame with one row per candidate, including validation
            status, flags, and priority score.
        """
        rows: list[dict] = []
        for candidate, validation in self.candidates:
            row = asdict(candidate)
            row["is_valid"] = validation.is_valid
            row["flags"] = "; ".join(validation.flags) if validation.flags else ""
            row["score"] = compute_score(candidate, validation)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    def summary(self) -> str:
        """Return a human-readable summary of the catalog.

        Returns:
            Multi-line string with candidate counts, highlights,
            and top candidates by score.
        """
        total = len(self.candidates)
        valid = len(self.get_valid())
        rejected = total - valid

        lines = [
            f"ExoHunter Candidate Catalog",
            f"{'=' * 50}",
            f"Total candidates:    {total}",
            f"Validated:           {valid}",
            f"Rejected:            {rejected}",
        ]

        if valid > 0:
            top = self.get_top(n=min(20, valid))
            lines.append(f"\nTop {len(top)} candidates by score:")
            lines.append(
                f"  {'TIC ID':<20s} {'Period (d)':>10s} "
                f"{'Depth (%)':>10s} {'SNR':>6s} {'Score':>7s}"
            )
            lines.append(f"  {'-'*55}")
            for candidate, validation, score in top:
                lines.append(
                    f"  {candidate.tic_id:<20s} {candidate.period:>10.4f} "
                    f"{candidate.depth * 100:>10.3f} {candidate.snr:>6.1f} "
                    f"{score:>7.1f}"
                )

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.candidates)

    def __repr__(self) -> str:
        return (
            f"CandidateCatalog(total={len(self)}, "
            f"valid={len(self.get_valid())}, "
            f"rejected={len(self.get_rejected())})"
        )
