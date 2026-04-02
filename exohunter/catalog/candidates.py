"""Candidate catalog management.

Stores, filters, and retrieves transit candidates discovered by the
pipeline.  The catalog is an in-memory list backed by optional
persistence to CSV.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from exohunter import config
from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult, validate_candidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


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
        logger.info(
            "Added candidate %s (P=%.4f d, valid=%s) — catalog size: %d",
            candidate.tic_id,
            candidate.period,
            validation.is_valid,
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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the catalog to a pandas DataFrame.

        Returns:
            A DataFrame with one row per candidate, including validation
            status and flags.
        """
        rows: list[dict] = []
        for candidate, validation in self.candidates:
            row = asdict(candidate)
            row["is_valid"] = validation.is_valid
            row["flags"] = "; ".join(validation.flags) if validation.flags else ""
            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a human-readable summary of the catalog.

        Returns:
            Multi-line string with candidate counts and highlights.
        """
        total = len(self.candidates)
        valid = len(self.get_valid())
        rejected = total - valid

        lines = [
            f"ExoHunter Candidate Catalog",
            f"{'=' * 40}",
            f"Total candidates:    {total}",
            f"Validated:           {valid}",
            f"Rejected:            {rejected}",
        ]

        if valid > 0:
            lines.append(f"\nValidated candidates:")
            for candidate in self.get_valid():
                lines.append(
                    f"  {candidate.tic_id}: P={candidate.period:.4f} d, "
                    f"depth={candidate.depth * 100:.3f}%, "
                    f"SNR={candidate.snr:.1f}"
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
