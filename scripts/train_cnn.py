#!/usr/bin/env python3
"""Train the CNN transit classifier on phase curves.

Generates synthetic phase curves from the Kepler KOI catalog
parameters, trains a 1D CNN, and saves the model.

Usage::

    # Train with defaults (requires Kepler KOI dataset)
    python scripts/train_cnn.py

    # Custom parameters
    python scripts/train_cnn.py --epochs 50 --batch-size 128

    # Use real Kepler light curves instead of synthetic
    python scripts/train_cnn.py --real-curves
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from exohunter.classification.cnn import (
    CLASS_LABELS,
    generate_training_phase_curves,
    save_cnn_model,
    train_cnn,
)
from exohunter.classification.datasets import prepare_kepler_koi
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Train the CNN and save it."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Train the CNN transit classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--real-curves",
        action="store_true",
        help=(
            "Download real Kepler light curves and generate phase curves "
            "from actual data instead of synthetic models. Much slower "
            "(downloads thousands of targets) but more realistic."
        ),
    )
    args = parser.parse_args()

    # Load catalog
    logger.info("Loading Kepler KOI catalog...")
    kepler_df = prepare_kepler_koi()
    logger.info("Catalog: %d labeled examples", len(kepler_df))

    if args.real_curves:
        logger.info("Real curve mode: downloading Kepler light curves...")
        logger.warning(
            "Real curve download is not yet implemented — falling back to "
            "synthetic. This is a contribution opportunity for students!"
        )
        # TODO: Students can implement real Kepler light curve download
        # using lightkurve and generate phase curves from actual data.

    # Generate synthetic phase curves
    logger.info("Generating synthetic phase curves from catalog parameters...")
    X_train, y_train = generate_training_phase_curves(kepler_df)
    logger.info(
        "Generated %d phase curves (%d bins each)",
        len(X_train), X_train.shape[1],
    )

    for i, label in enumerate(CLASS_LABELS):
        count = int(np.sum(y_train == i))
        logger.info("  %s: %d", label, count)

    # Train
    model = train_cnn(
        X_train, y_train,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Save
    path = save_cnn_model(model)
    logger.info("CNN model saved to: %s", path)


if __name__ == "__main__":
    main()
