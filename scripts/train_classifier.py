#!/usr/bin/env python3
"""Train the transit candidate classifier.

Trains a Random Forest on the Kepler KOI dataset and optionally
validates on the ExoFOP-TESS TOI catalog. Saves the trained model
to ``data/models/transit_classifier.joblib``.

Usage::

    # Train (requires datasets — run download_training_data.py first)
    python scripts/train_classifier.py

    # Train with custom parameters
    python scripts/train_classifier.py --n-estimators 500 --max-depth 30

    # Train and validate on TESS data
    python scripts/train_classifier.py --validate-tess
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import classification_report

from exohunter.classification.datasets import prepare_exofop_toi, prepare_kepler_koi
from exohunter.classification.features import FEATURE_COLUMNS
from exohunter.classification.model import (
    CLASS_LABELS,
    classify_candidates,
    save_model,
    train,
)
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Train the classifier and optionally validate on TESS data."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Train the transit candidate classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the Random Forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth (0 for unlimited)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--validate-tess",
        action="store_true",
        help="Also validate on the ExoFOP-TESS TOI dataset",
    )
    args = parser.parse_args()

    max_depth = args.max_depth if args.max_depth > 0 else None

    # Load training data
    logger.info("Loading Kepler KOI training data...")
    kepler_df = prepare_kepler_koi()
    logger.info("Training set: %d examples", len(kepler_df))

    # Train
    pipeline = train(
        kepler_df,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        cv_folds=args.cv_folds,
    )

    # Feature importances
    rf = pipeline.named_steps["classifier"]
    importances = rf.feature_importances_
    logger.info("Feature importances:")
    for name, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        logger.info("  %-25s %.4f", name, imp)

    # Save model
    model_path = save_model(pipeline)
    logger.info("Model saved to: %s", model_path)

    # Optional TESS validation
    if args.validate_tess:
        logger.info("=" * 60)
        logger.info("Validating on ExoFOP-TESS TOI dataset...")
        try:
            toi_df = prepare_exofop_toi()
            # ExoFOP doesn't have eclipsing_binary labels, so we
            # map our 3-class predictions to a binary for comparison
            # (planet vs. not-planet)
            results = classify_candidates(pipeline, toi_df)

            # Compare ML predictions with ExoFOP labels
            y_true = toi_df["label"].values
            y_pred = results["ml_class"].values

            # Map to binary for comparison (ExoFOP only has planet/FP)
            y_true_binary = np.where(y_true == "planet", "planet", "not_planet")
            y_pred_binary = np.where(y_pred == "planet", "planet", "not_planet")

            report = classification_report(
                y_true_binary, y_pred_binary, zero_division=0
            )
            logger.info("TESS validation (binary: planet vs not_planet):\n%s", report)

        except FileNotFoundError:
            logger.warning(
                "ExoFOP TOI dataset not found — skipping TESS validation. "
                "Run: python scripts/download_training_data.py"
            )


if __name__ == "__main__":
    main()
