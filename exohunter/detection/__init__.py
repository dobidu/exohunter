"""Transit detection layer — BLS periodogram and candidate validation."""

from exohunter.detection.bls import (
    TransitCandidate,
    run_bls_lightkurve,
    run_bls_numba,
    run_iterative_bls,
)
from exohunter.detection.validator import validate_candidate

__all__ = [
    "TransitCandidate",
    "run_bls_lightkurve",
    "run_bls_numba",
    "run_iterative_bls",
    "validate_candidate",
]
