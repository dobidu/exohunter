"""Feature extraction for transit candidate classification.

Converts ExoHunter ``TransitCandidate`` objects and
``ValidationResult`` objects into the feature vector expected
by the trained Random Forest classifier.

The feature schema matches the Kepler KOI training data:
    - period, depth, duration (from BLS detection)
    - snr (from BLS or lightkurve stats)
    - impact_param (estimated from V-shape / ingress fraction)
    - stellar_teff, stellar_logg, stellar_radius (from TIC metadata)
    - duration_period_ratio (engineered)
    - depth_log (engineered)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# The 10 features used by the classifier, in the exact order
# expected by the trained model.
FEATURE_COLUMNS: list[str] = [
    "period",
    "depth",
    "duration",
    "snr",
    "impact_param",
    "stellar_teff",
    "stellar_logg",
    "stellar_radius",
    "duration_period_ratio",
    "depth_log",
]


def candidate_to_features(
    candidate: TransitCandidate,
    validation: ValidationResult | None = None,
    stellar_params: dict | None = None,
) -> dict[str, float]:
    """Extract the ML feature vector from a single candidate.

    Args:
        candidate: The transit candidate from BLS detection.
        validation: The validation result (used for V-shape estimate).
            If ``None``, impact parameter defaults to 0.0.
        stellar_params: Optional dict with ``teff``, ``logg``,
            ``radius`` keys (from TIC catalog or FITS metadata).

    Returns:
        A dict mapping feature names to values.
    """
    depth = candidate.depth
    period = candidate.period
    duration = candidate.duration

    # Impact parameter estimate from V-shape test:
    # V-shape passed → box-like → low impact (face-on) → ~0.2
    # V-shape failed → V-like → high impact (grazing) → ~0.7
    if validation is not None:
        v_shape_passed = validation.tests.get("v_shape", True)
        impact_param = 0.2 if v_shape_passed else 0.7
    else:
        impact_param = 0.0

    # Stellar parameters (default to solar values if unavailable)
    if stellar_params is None:
        stellar_params = {}
    stellar_teff = stellar_params.get("teff", 5778.0)
    stellar_logg = stellar_params.get("logg", 4.44)
    stellar_radius = stellar_params.get("radius", 1.0)

    return {
        "period": period,
        "depth": depth,
        "duration": duration,
        "snr": candidate.snr,
        "impact_param": impact_param,
        "stellar_teff": stellar_teff,
        "stellar_logg": stellar_logg,
        "stellar_radius": stellar_radius,
        "duration_period_ratio": duration / period if period > 0 else 0.0,
        "depth_log": float(np.log10(max(depth, 1e-10))),
    }


def candidates_to_dataframe(
    candidates: list[tuple[TransitCandidate, ValidationResult | None]],
) -> pd.DataFrame:
    """Convert a batch of candidates to a feature DataFrame.

    Args:
        candidates: List of ``(candidate, validation)`` tuples.

    Returns:
        A DataFrame with one row per candidate and columns matching
        ``FEATURE_COLUMNS``.
    """
    rows = [candidate_to_features(c, v) for c, v in candidates]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df
