#!/usr/bin/env python3
"""CLI script to run the full ExoHunter pipeline.

Usage::

    # Process a single target
    python scripts/run_pipeline.py --tic "TIC 150428135"

    # Process all targets in a TESS sector (limit to first N)
    python scripts/run_pipeline.py --sector 1 --limit 10

    # Custom period search range
    python scripts/run_pipeline.py --tic "TIC 150428135" --min-period 1.0 --max-period 40.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so this script works without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exohunter import config
from exohunter.catalog.candidates import CandidateCatalog
from exohunter.catalog.crossmatch import crossmatch_candidate
from exohunter.catalog.export import export_to_csv
from exohunter.detection.bls import TransitCandidate, run_bls_lightkurve
from exohunter.detection.validator import validate_candidate
from exohunter.ingestion.downloader import download_batch, download_lightcurve, search_targets
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Exoplanet Transit Detection Pipeline",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tic",
        type=str,
        help="TIC ID of a single target (e.g. 'TIC 150428135')",
    )
    group.add_argument(
        "--sector",
        type=int,
        help="TESS sector number to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of targets to process from a sector",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=config.BLS_MIN_PERIOD_DAYS,
        help="Minimum BLS search period in days",
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=config.BLS_MAX_PERIOD_DAYS,
        help="Maximum BLS search period in days",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_DOWNLOAD_WORKERS,
        help="Number of download threads",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline."""
    args = parse_args()
    catalog = CandidateCatalog()

    # Step 1: Identify targets
    if args.tic:
        tic_ids = [args.tic]
    else:
        logger.info("Searching for targets in TESS sector %d", args.sector)
        tic_ids = search_targets(args.sector, limit=args.limit)

    logger.info("Pipeline will process %d target(s)", len(tic_ids))

    # Step 2: Download light curves
    if len(tic_ids) == 1:
        lc = download_lightcurve(tic_ids[0])
        light_curves = [lc] if lc is not None else []
    else:
        light_curves = download_batch(tic_ids, max_workers=args.workers)

    if not light_curves:
        logger.error("No light curves downloaded — exiting")
        sys.exit(1)

    # Step 3: Preprocess and detect transits
    for i, lc in enumerate(light_curves):
        tic_id = tic_ids[i] if i < len(tic_ids) else f"target_{i}"

        # Preprocess
        processed = preprocess_single(lc, tic_id=tic_id)

        # Run BLS detection
        lc_for_bls = processed.to_lightcurve()
        candidate = run_bls_lightkurve(
            lc_for_bls,
            tic_id=tic_id,
            min_period=args.min_period,
            max_period=args.max_period,
        )

        if candidate is None:
            logger.info("No transit detected for %s", tic_id)
            continue

        # Validate
        validation = validate_candidate(
            candidate,
            time=processed.time,
            flux=processed.flux,
        )

        # Add to catalog
        catalog.add(candidate, validation)

        # Cross-match with TOI catalog
        xmatch = crossmatch_candidate(candidate)
        logger.info(
            "Cross-match classification: %s%s",
            xmatch.match_class.value,
            f" ({xmatch.catalog_name})" if xmatch.catalog_name else "",
        )

    # Step 4: Report results
    print("\n" + catalog.summary())

    # Step 5: Export
    csv_path = export_to_csv(catalog)
    print(f"\nResults exported to: {csv_path}")


if __name__ == "__main__":
    main()
