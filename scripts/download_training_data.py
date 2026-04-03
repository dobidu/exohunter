#!/usr/bin/env python3
"""Download training datasets for the ML classifier.

Reads dataset URLs from ``data/datasets_sources.json`` and downloads
the Kepler KOI cumulative table and ExoFOP-TESS TOI catalog.

Usage::

    python scripts/download_training_data.py
    python scripts/download_training_data.py --force   # re-download
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exohunter.classification.datasets import (
    download_all,
    prepare_exofop_toi,
    prepare_kepler_koi,
)
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Download and validate all training datasets."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Download ML training datasets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    logger.info("Downloading training datasets...")
    paths = download_all(force=args.force)

    for name, path in paths.items():
        logger.info("  %s: %s", name, path)

    # Validate by loading and preparing each dataset
    logger.info("Validating Kepler KOI dataset...")
    kepler_df = prepare_kepler_koi()
    logger.info("  Kepler KOI: %d labeled examples, %d features",
                len(kepler_df), len(kepler_df.columns) - 1)

    logger.info("Validating ExoFOP TOI dataset...")
    toi_df = prepare_exofop_toi()
    logger.info("  ExoFOP TOI: %d labeled examples, %d features",
                len(toi_df), len(toi_df.columns) - 1)

    logger.info("All datasets downloaded and validated.")


if __name__ == "__main__":
    main()
