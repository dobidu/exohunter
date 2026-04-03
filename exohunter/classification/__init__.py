"""ML-based transit candidate classification.

Provides a Random Forest classifier trained on the Kepler KOI dataset
to distinguish genuine exoplanet transits from false positives and
eclipsing binaries.
"""

from exohunter.classification.model import classify_candidates, load_model

__all__ = ["classify_candidates", "load_model"]
