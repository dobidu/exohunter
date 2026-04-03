#!/usr/bin/env python3
"""Batch processing of an entire TESS sector.

Downloads, preprocesses, and runs BLS transit detection on all targets
in a given TESS sector, filtering by magnitude to focus on stars bright
enough for good photometry but faint enough to be under-studied.

Usage::

    # Process sector 56 (all targets with Tmag 10–14)
    python scripts/run_batch.py --sector 56

    # Limit to first 20 targets (for testing)
    python scripts/run_batch.py --sector 56 --limit 20

    # Custom magnitude range and period search
    python scripts/run_batch.py --sector 56 --mag-min 9 --mag-max 12 --max-period 30
"""

import argparse
import sys
import time as time_module
from pathlib import Path

# Add project root to path so this script works without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from exohunter import config
from exohunter.catalog.candidates import CandidateCatalog
from exohunter.catalog.crossmatch import crossmatch_candidate
from exohunter.detection.bls import run_bls_lightkurve
from exohunter.detection.validator import validate_candidate
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.utils.logging import get_logger
from exohunter.utils.parallel import run_parallel_threads

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Target discovery with magnitude filtering
# ---------------------------------------------------------------------------

def search_sector_with_magnitudes(
    sector: int,
    mag_min: float = 10.0,
    mag_max: float = 14.0,
    limit: int | None = None,
) -> list[dict]:
    """Search for targets in a TESS sector and filter by magnitude.

    Queries MAST for all 2-minute cadence observations in the sector,
    then filters by TESS magnitude using a single TIC catalog query
    with the Tmag range pre-applied on the server side.

    The magnitude window ``[10, 14]`` targets stars that are:
        - Bright enough (Tmag < 14) for transit photometry with
          good signal-to-noise.
        - Faint enough (Tmag > 10) to be less thoroughly studied
          by ground-based follow-up, increasing the chance of
          finding new candidates.

    Args:
        sector: TESS sector number.
        mag_min: Minimum TESS magnitude (brighter limit).
        mag_max: Maximum TESS magnitude (fainter limit).
        limit: Maximum number of targets to return after filtering.

    Returns:
        List of dicts with ``tic_id`` (str) and ``tmag`` (float).
    """
    logger.info(
        "Searching sector %d for targets with Tmag %.1f–%.1f",
        sector, mag_min, mag_max,
    )

    from astroquery.mast import Observations

    # Step 1: Get all 2-min cadence observations in the sector
    obs_table = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
        t_exptime=[100, 200],
    )

    if obs_table is None or len(obs_table) == 0:
        logger.warning("No observations found in sector %d", sector)
        return []

    # Extract unique TIC IDs
    unique_tics: set[str] = set()
    for row in obs_table:
        raw_name = str(row["target_name"]).strip()
        unique_tics.add(raw_name.replace("TIC ", "").replace("TIC", "").strip())

    all_numeric_ids = list(unique_tics)
    logger.info("Found %d unique targets in sector %d", len(all_numeric_ids), sector)

    # Step 2: Query TIC catalog with the magnitude filter applied
    #         server-side so we only transfer matching rows.
    targets = _query_tic_magnitudes(all_numeric_ids, mag_min, mag_max)

    logger.info(
        "After magnitude filter (%.1f–%.1f): %d / %d targets",
        mag_min, mag_max, len(targets), len(all_numeric_ids),
    )

    if limit is not None:
        targets = targets[:limit]
        logger.info("Limited to %d targets", limit)

    return targets


def _query_tic_magnitudes(
    numeric_ids: list[str],
    mag_min: float,
    mag_max: float,
) -> list[dict]:
    """Query the TIC catalog for TESS magnitudes, filtering server-side.

    Sends the full list of numeric TIC IDs to MAST with the Tmag
    constraint, so only matching rows are returned — much faster
    than downloading all magnitudes and filtering locally.

    Args:
        numeric_ids: List of numeric TIC IDs (no "TIC " prefix).
        mag_min: Minimum (brightest) Tmag.
        mag_max: Maximum (faintest) Tmag.

    Returns:
        Filtered list of dicts with ``tic_id`` and ``tmag``,
        sorted by magnitude (brightest first).
    """
    try:
        from astroquery.mast import Catalogs

        results: list[dict] = []

        # Query in batches of 5000 — MAST handles this size well
        # and the Tmag filter is applied server-side so the response
        # is small even for large batches.
        batch_size = 5000
        n_total = len(numeric_ids)
        for i in range(0, n_total, batch_size):
            batch = numeric_ids[i : i + batch_size]
            batch_end = min(i + batch_size, n_total)
            logger.info(
                "Querying TIC magnitudes: %d–%d / %d",
                i + 1, batch_end, n_total,
            )

            catalog_data = Catalogs.query_criteria(
                catalog="TIC",
                ID=batch,
                Tmag=[mag_min, mag_max],
            )

            if catalog_data is not None and len(catalog_data) > 0:
                for row in catalog_data:
                    tmag_val = row["Tmag"]
                    if tmag_val is None or np.ma.is_masked(tmag_val):
                        continue
                    tmag = float(tmag_val)
                    if mag_min <= tmag <= mag_max:
                        results.append({
                            "tic_id": f"TIC {row['ID']}",
                            "tmag": tmag,
                        })

        # Sort by magnitude (brightest first — best photometry)
        results.sort(key=lambda x: x["tmag"])
        return results

    except Exception as exc:
        logger.warning(
            "TIC catalog query failed (%s) — returning unfiltered list", exc
        )
        return [{"tic_id": f"TIC {n}", "tmag": float("nan")} for n in numeric_ids]


# ---------------------------------------------------------------------------
# Main batch driver
# ---------------------------------------------------------------------------

def run_batch(
    sector: int,
    mag_min: float = 10.0,
    mag_max: float = 14.0,
    limit: int | None = None,
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
    download_workers: int = config.DEFAULT_DOWNLOAD_WORKERS,
) -> tuple[CandidateCatalog, pd.DataFrame]:
    """Run the full ExoHunter pipeline on a TESS sector.

    Stages:
        1. Search sector and filter by Tmag
        2. Download light curves (ThreadPoolExecutor, I/O-bound)
        3. For each downloaded curve: preprocess → BLS → validate
        4. Collect validated candidates into a catalog

    Args:
        sector: TESS sector number.
        mag_min: Minimum TESS magnitude.
        mag_max: Maximum TESS magnitude.
        limit: Max number of targets to process.
        min_period: BLS search minimum period (days).
        max_period: BLS search maximum period (days).
        download_workers: Number of concurrent download threads.

    Returns:
        Tuple of ``(catalog, summary_df)`` where ``summary_df``
        contains one row per processed target with its status.
    """
    batch_start = time_module.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1: Discover targets
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  ExoHunter Batch Processing — TESS Sector %d", sector)
    logger.info("  Magnitude range: %.1f–%.1f Tmag", mag_min, mag_max)
    logger.info("  Period search: %.1f–%.1f days", min_period, max_period)
    logger.info("=" * 70)

    targets = search_sector_with_magnitudes(
        sector=sector,
        mag_min=mag_min,
        mag_max=mag_max,
        limit=limit,
    )

    if not targets:
        logger.warning("No targets found matching criteria. Exiting.")
        return CandidateCatalog(), pd.DataFrame()

    tic_ids = [t["tic_id"] for t in targets]
    n_targets = len(tic_ids)
    logger.info("[Stage 1] %d targets to process", n_targets)

    # ------------------------------------------------------------------
    # Stage 2: Download light curves (concurrent threads)
    # ------------------------------------------------------------------
    logger.info("[Stage 2] Downloading light curves (%d threads)...", download_workers)

    downloaded: dict[str, object] = {}

    def _download_one(tic_id: str):
        return download_lightcurve(tic_id)

    # Save references to stdout/stderr before threaded downloads.
    # lightkurve's internal file operations can corrupt sys.stdout
    # when run concurrently — we restore them afterwards.
    _saved_stdout = sys.stdout
    _saved_stderr = sys.stderr

    raw_results = run_parallel_threads(
        func=_download_one,
        items=tic_ids,
        max_workers=download_workers,
        description="Downloading",
    )

    # Restore stdout/stderr in case threaded downloads corrupted them
    sys.stdout = _saved_stdout
    sys.stderr = _saved_stderr

    for i, lc in enumerate(raw_results):
        if lc is not None:
            downloaded[tic_ids[i]] = lc

    n_downloaded = len(downloaded)
    logger.info("Downloaded %d / %d light curves", n_downloaded, n_targets)

    if not downloaded:
        logger.error("No light curves downloaded. Exiting.")
        return CandidateCatalog(), pd.DataFrame()

    # ------------------------------------------------------------------
    # Stage 3 + 4: Preprocess → BLS → validate (sequential with tqdm)
    #
    # Sequential processing: each target's pipeline includes BLS (which
    # uses astropy internally) and the overhead of pickling LightCurve
    # objects across processes outweighs the parallelism gain at typical
    # sector sizes (~100-200 targets after magnitude filtering).
    # ------------------------------------------------------------------
    logger.info("[Stage 3] Processing: preprocess → BLS → validate")

    catalog = CandidateCatalog()
    summary_rows: list[dict] = []
    n_processed = 0
    n_failed = 0

    pbar = tqdm(
        downloaded.items(),
        total=n_downloaded,
        desc="Processing",
        unit="star",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        file=sys.stderr,
    )

    for tic_id, light_curve in pbar:
        pbar.set_postfix_str(tic_id, refresh=False)

        try:
            # Preprocess
            processed = preprocess_single(light_curve, tic_id=tic_id)

            # BLS detection
            lc_for_bls = processed.to_lightcurve()
            candidate = run_bls_lightkurve(
                lc_for_bls,
                tic_id=tic_id,
                min_period=min_period,
                max_period=max_period,
            )

            if candidate is None:
                summary_rows.append({
                    "tic_id": tic_id,
                    "status": "no_signal",
                    "period": None,
                    "depth": None,
                    "snr": None,
                })
                n_processed += 1
                continue

            # Validate
            validation = validate_candidate(
                candidate,
                time=processed.time,
                flux=processed.flux,
            )

            # Determine status via cross-matching
            status = "below_snr"
            xmatch_class = ""
            if candidate.snr >= config.MIN_SNR:
                status = "candidate"
                if validation.is_valid:
                    catalog.add(candidate, validation)

                    # Cross-match against TOI catalog
                    xmatch = crossmatch_candidate(candidate)
                    xmatch_class = xmatch.match_class.value
                    status = xmatch_class.lower()

            summary_rows.append({
                "tic_id": tic_id,
                "status": status,
                "xmatch_class": xmatch_class,
                "period": candidate.period,
                "depth": candidate.depth,
                "snr": candidate.snr,
                "duration": candidate.duration,
                "n_transits": candidate.n_transits,
                "is_valid": validation.is_valid,
                "flags": "; ".join(validation.flags),
            })
            n_processed += 1

        except Exception:
            logger.exception("Failed to process %s", tic_id)
            summary_rows.append({
                "tic_id": tic_id,
                "status": "error",
                "period": None,
                "depth": None,
                "snr": None,
            })
            n_failed += 1

    pbar.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    batch_elapsed = time_module.perf_counter() - batch_start
    summary_df = pd.DataFrame(summary_rows)

    _print_summary(
        sector=sector,
        n_targets=n_targets,
        n_downloaded=n_downloaded,
        n_processed=n_processed,
        n_failed=n_failed,
        catalog=catalog,
        summary_df=summary_df,
        elapsed=batch_elapsed,
    )

    return catalog, summary_df


def _print_summary(
    sector: int,
    n_targets: int,
    n_downloaded: int,
    n_processed: int,
    n_failed: int,
    catalog: CandidateCatalog,
    summary_df: pd.DataFrame,
    elapsed: float,
) -> None:
    """Log the end-of-batch summary report."""
    n_valid = len(catalog.get_valid())

    if elapsed < 60:
        time_str = f"{elapsed:.1f}s"
    else:
        minutes, seconds = divmod(elapsed, 60)
        time_str = f"{int(minutes)}m {seconds:.0f}s"

    lines = [
        "",
        "=" * 70,
        f"  Batch Results — TESS Sector {sector}",
        "=" * 70,
        f"  Targets found:       {n_targets}",
        f"  Downloaded:          {n_downloaded}",
        f"  Processed:           {n_processed}",
        f"  Failed:              {n_failed}",
        f"  Candidates (SNR>7):  {n_valid}",
        f"  Total time:          {time_str}",
    ]

    if n_processed > 0:
        lines.append(f"  Time per target:     {elapsed / n_processed:.1f}s")

    # Status breakdown
    if not summary_df.empty and "status" in summary_df.columns:
        lines.append("")
        lines.append("  Status breakdown:")
        for status, count in summary_df["status"].value_counts().items():
            lines.append(f"    {status:20s}  {count}")

    # Top 10 candidates by SNR
    if n_valid > 0:
        top_n = min(10, n_valid)
        lines.append("")
        lines.append(f"  Top {top_n} candidates by SNR:")
        lines.append(f"  {'TIC ID':<20s} {'Period (d)':>10s} {'Depth (%)':>10s} {'SNR':>8s}")
        lines.append(f"  {'-'*50}")

        valid_candidates = catalog.get_valid()
        top_candidates = sorted(valid_candidates, key=lambda c: c.snr, reverse=True)[:10]
        for c in top_candidates:
            lines.append(
                f"  {c.tic_id:<20s} {c.period:>10.4f} "
                f"{c.depth * 100:>10.4f} {c.snr:>8.1f}"
            )

    lines.append("=" * 70)
    lines.append("")

    # Write the full summary as a single log message to avoid interleaving
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Batch processing of a TESS sector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sector",
        type=int,
        required=True,
        help="TESS sector number to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of targets to process (for testing)",
    )
    parser.add_argument(
        "--mag-min",
        type=float,
        default=10.0,
        help="Minimum TESS magnitude (brightest)",
    )
    parser.add_argument(
        "--mag-max",
        type=float,
        default=14.0,
        help="Maximum TESS magnitude (faintest)",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=config.BLS_MIN_PERIOD_DAYS,
        help="Minimum BLS search period (days)",
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=config.BLS_MAX_PERIOD_DAYS,
        help="Maximum BLS search period (days)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_DOWNLOAD_WORKERS,
        help="Number of concurrent download threads",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for batch processing."""
    args = parse_args()

    catalog, summary_df = run_batch(
        sector=args.sector,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        limit=args.limit,
        min_period=args.min_period,
        max_period=args.max_period,
        download_workers=args.workers,
    )

    # Save results
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"sector_{args.sector:02d}.csv"
    if not summary_df.empty:
        summary_df.to_csv(csv_path, index=False)
        logger.info("Results saved to: %s", csv_path)
    else:
        logger.info("No results to save.")

    # Also export the validated candidate catalog
    if len(catalog) > 0:
        from exohunter.catalog.export import export_to_csv
        catalog_path = results_dir / f"sector_{args.sector:02d}_candidates.csv"
        export_to_csv(catalog, output_path=catalog_path)
        logger.info("Candidate catalog saved to: %s", catalog_path)


if __name__ == "__main__":
    main()
