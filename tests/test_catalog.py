"""Tests for the candidate catalog, scoring, and cross-matching.

Verifies:
    - Candidate scoring formula produces correct values
    - Catalog ranking sorts by score
    - DataFrame export includes all fields
    - Cross-match classifies correctly against built-in reference table
    - All four MatchClass categories are reachable
"""

import numpy as np
import pytest

from exohunter.catalog.candidates import CandidateCatalog, compute_score
from exohunter.catalog.crossmatch import (
    CrossMatchResult,
    MatchClass,
    _extract_tic_number,
    crossmatch_batch,
    crossmatch_candidate,
)
from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult
from tests.conftest import make_candidate, make_validation


# =========================================================================
# Scoring
# =========================================================================

class TestComputeScore:
    """Test the candidate scoring formula."""

    def test_base_score_equals_snr(self) -> None:
        """With no penalties, score must equal SNR exactly."""
        c = make_candidate(snr=15.0, depth=0.005)
        v = make_validation(v_shape_pass=True)
        assert compute_score(c, v) == 15.0

    def test_v_shape_penalty(self) -> None:
        """V-shaped transit must reduce score by 50%."""
        c = make_candidate(snr=20.0, depth=0.005)
        v_good = make_validation(v_shape_pass=True)
        v_bad = make_validation(v_shape_pass=False)

        assert compute_score(c, v_good) == 20.0
        assert compute_score(c, v_bad) == 10.0

    def test_depth_penalty(self) -> None:
        """Depth >= 2% must reduce score by 30%."""
        c_shallow = make_candidate(snr=10.0, depth=0.01)  # 1% — no penalty
        c_deep = make_candidate(snr=10.0, depth=0.03)     # 3% — penalty

        v = make_validation(v_shape_pass=True)

        assert compute_score(c_shallow, v) == 10.0
        assert compute_score(c_deep, v) == 7.0

    def test_both_penalties(self) -> None:
        """V-shape + depth penalties must multiply: 0.5 × 0.7 = 0.35."""
        c = make_candidate(snr=20.0, depth=0.04)
        v = make_validation(v_shape_pass=False)

        assert compute_score(c, v) == 7.0  # 20 × 0.5 × 0.7

    def test_score_is_non_negative(self) -> None:
        """Score must never be negative, even with SNR=0."""
        c = make_candidate(snr=0.0, depth=0.001)
        v = make_validation()
        assert compute_score(c, v) >= 0.0

    def test_depth_boundary_at_two_percent(self) -> None:
        """Depth penalty boundary: 1.99% → no penalty, 2.0% → penalty."""
        v = make_validation()

        c_under = make_candidate(snr=10.0, depth=0.0199)
        c_at = make_candidate(snr=10.0, depth=0.02)

        assert compute_score(c_under, v) == 10.0
        assert compute_score(c_at, v) == 7.0


# =========================================================================
# Catalog
# =========================================================================

class TestCandidateCatalog:
    """Test the CandidateCatalog class."""

    def test_add_and_len(self) -> None:
        """Adding candidates must increase catalog length."""
        catalog = CandidateCatalog()
        assert len(catalog) == 0

        catalog.add(make_candidate(tic_id="A"), make_validation())
        assert len(catalog) == 1

        catalog.add(make_candidate(tic_id="B"), make_validation())
        assert len(catalog) == 2

    def test_get_valid_filters_correctly(self) -> None:
        """get_valid must only return candidates with is_valid=True."""
        catalog = CandidateCatalog()

        catalog.add(make_candidate(tic_id="GOOD"), make_validation(is_valid=True))
        catalog.add(make_candidate(tic_id="BAD"), make_validation(is_valid=False))
        catalog.add(make_candidate(tic_id="GOOD2"), make_validation(is_valid=True))

        valid = catalog.get_valid()
        rejected = catalog.get_rejected()

        assert len(valid) == 2
        assert len(rejected) == 1
        assert all(c.tic_id.startswith("GOOD") for c in valid)
        assert rejected[0].tic_id == "BAD"

    def test_get_ranked_sorts_by_score(self) -> None:
        """get_ranked must return candidates sorted by descending score."""
        catalog = CandidateCatalog()

        catalog.add(make_candidate(tic_id="LOW", snr=5.0), make_validation())
        catalog.add(make_candidate(tic_id="HIGH", snr=25.0), make_validation())
        catalog.add(make_candidate(tic_id="MID", snr=12.0), make_validation())

        ranked = catalog.get_ranked()
        scores = [s for _, _, s in ranked]

        assert scores == sorted(scores, reverse=True)
        assert ranked[0][0].tic_id == "HIGH"
        assert ranked[-1][0].tic_id == "LOW"

    def test_get_top_limits_results(self) -> None:
        """get_top(n) must return at most n candidates."""
        catalog = CandidateCatalog()
        for i in range(10):
            catalog.add(make_candidate(tic_id=f"C{i}", snr=float(i)), make_validation())

        top3 = catalog.get_top(n=3)
        assert len(top3) == 3
        # Best should be the highest SNR
        assert top3[0][0].tic_id == "C9"

    def test_to_dataframe_columns(self) -> None:
        """to_dataframe must include score, is_valid, and flags columns."""
        catalog = CandidateCatalog()
        catalog.add(
            make_candidate(tic_id="DF_TEST", snr=10.0, depth=0.005, period=8.0),
            make_validation(),
        )

        df = catalog.to_dataframe()

        required_columns = {"tic_id", "period", "depth", "snr", "score", "is_valid", "flags"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing columns: {required_columns - set(df.columns)}"
        )

        row = df.iloc[0]
        assert row["tic_id"] == "DF_TEST"
        assert row["score"] == 10.0
        assert row["is_valid"] == True

    def test_to_dataframe_sorted_by_score(self) -> None:
        """DataFrame must be sorted by score descending."""
        catalog = CandidateCatalog()
        catalog.add(make_candidate(tic_id="A", snr=5.0), make_validation())
        catalog.add(make_candidate(tic_id="B", snr=20.0), make_validation())
        catalog.add(make_candidate(tic_id="C", snr=10.0), make_validation())

        df = catalog.to_dataframe()
        assert list(df["tic_id"]) == ["B", "C", "A"]

    def test_summary_contains_header(self) -> None:
        """summary() must contain the catalog header."""
        catalog = CandidateCatalog()
        catalog.add(make_candidate(), make_validation())

        text = catalog.summary()
        assert "ExoHunter Candidate Catalog" in text
        assert "Total candidates:" in text

    def test_empty_catalog(self) -> None:
        """An empty catalog must not crash on any operation."""
        catalog = CandidateCatalog()

        assert len(catalog) == 0
        assert catalog.get_valid() == []
        assert catalog.get_rejected() == []
        assert catalog.get_ranked() == []
        assert catalog.get_top() == []
        assert catalog.to_dataframe().empty
        assert "Total candidates:    0" in catalog.summary()


# =========================================================================
# Cross-matching
# =========================================================================

class TestTicIdParsing:
    """Test the TIC ID extraction helper."""

    def test_with_prefix(self) -> None:
        assert _extract_tic_number("TIC 150428135") == 150428135

    def test_without_prefix(self) -> None:
        assert _extract_tic_number("150428135") == 150428135

    def test_with_tic_no_space(self) -> None:
        assert _extract_tic_number("TIC150428135") == 150428135

    def test_invalid_returns_none(self) -> None:
        assert _extract_tic_number("not-a-number") is None

    def test_empty_string(self) -> None:
        assert _extract_tic_number("") is None


class TestCrossMatchClassification:
    """Test the 4-tier cross-match classification using built-in reference data."""

    def test_known_match(self) -> None:
        """TOI-700 b at P=9.977 d must classify as KNOWN_MATCH."""
        c = make_candidate(tic_id="TIC 150428135", period=9.977)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.KNOWN_MATCH
        assert result.match_found is True
        assert "TOI-700" in result.catalog_name
        assert result.period_difference < 0.1

    def test_known_match_within_tolerance(self) -> None:
        """A period within 0.1 d of TOI-700 b must still be KNOWN_MATCH."""
        c = make_candidate(tic_id="TIC 150428135", period=9.90)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.KNOWN_MATCH

    def test_harmonic_double(self) -> None:
        """P=19.95 d ≈ 2× TOI-700 b must classify as HARMONIC."""
        c = make_candidate(tic_id="TIC 150428135", period=19.95)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.HARMONIC

    def test_harmonic_half(self) -> None:
        """P=4.99 d ≈ 0.5× TOI-700 b must classify as HARMONIC."""
        c = make_candidate(tic_id="TIC 150428135", period=4.99)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.HARMONIC

    def test_known_toi_different_period(self) -> None:
        """A known TIC with a non-matching, non-harmonic period → KNOWN_TOI."""
        c = make_candidate(tic_id="TIC 150428135", period=6.0)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.KNOWN_TOI
        assert result.match_found is True

    def test_new_candidate(self) -> None:
        """An unknown TIC must classify as NEW_CANDIDATE."""
        c = make_candidate(tic_id="TIC 999999999", period=3.5)
        result = crossmatch_candidate(c)

        assert result.match_class == MatchClass.NEW_CANDIDATE
        assert result.match_found is False
        assert result.catalog_name == ""

    def test_status_field_syncs_with_match_class(self) -> None:
        """The legacy status field must match the MatchClass value."""
        c = make_candidate(tic_id="TIC 999999999", period=3.5)
        result = crossmatch_candidate(c)

        assert result.status == "NEW_CANDIDATE"

    def test_crossmatch_batch(self) -> None:
        """crossmatch_batch must process a list and preserve order."""
        candidates = [
            make_candidate(tic_id="TIC 150428135", period=9.977),
            make_candidate(tic_id="TIC 999999999", period=5.0),
        ]
        results = crossmatch_batch(candidates)

        assert len(results) == 2
        assert results[0].match_class == MatchClass.KNOWN_MATCH
        assert results[1].match_class == MatchClass.NEW_CANDIDATE
