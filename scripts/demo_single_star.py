#!/usr/bin/env python3
"""Demo: process TOI-700 (TIC 150428135) end-to-end.

TOI-700 is a well-studied TESS target with three confirmed planets:
    - TOI-700 b: period ≈  9.98 days
    - TOI-700 c: period ≈ 16.05 days
    - TOI-700 d: period ≈ 37.42 days (in the habitable zone!)

This script demonstrates the complete ExoHunter pipeline:
    1. Download data from MAST
    2. Preprocess (clean, normalize, detrend)
    3. Run BLS transit search
    4. Validate candidates
    5. Cross-match with known catalogs
    6. Generate output plots

Usage::

    python scripts/demo_single_star.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from exohunter import config
from exohunter.catalog.candidates import CandidateCatalog
from exohunter.catalog.crossmatch import crossmatch_candidate
from exohunter.catalog.export import export_to_csv
from exohunter.detection.bls import run_bls_lightkurve
from exohunter.detection.model import bin_phase_curve, phase_fold, transit_model_from_candidate
from exohunter.detection.validator import validate_candidate
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# Target: TOI-700 — a nearby M-dwarf with confirmed transiting planets
TARGET_TIC = "TIC 150428135"
TARGET_NAME = "TOI-700"


def main() -> None:
    """Run the full pipeline on TOI-700."""
    print(f"\n{'='*60}")
    print(f"  ExoHunter Demo — {TARGET_NAME} ({TARGET_TIC})")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Step 1: Download
    # ---------------------------------------------------------------
    print("[1/5] Downloading light curve from MAST...")
    light_curve = download_lightcurve(TARGET_TIC)

    if light_curve is None:
        print("ERROR: Could not download data for TOI-700.")
        print("Check your internet connection and try again.")
        sys.exit(1)

    print(f"      Downloaded {len(light_curve)} cadences")

    # ---------------------------------------------------------------
    # Step 2: Preprocess
    # ---------------------------------------------------------------
    print("[2/5] Preprocessing light curve...")
    processed = preprocess_single(light_curve, tic_id=TARGET_TIC)
    print(f"      {len(processed.flux)} cadences after cleaning, CDPP={processed.cdpp:.1f} ppm")

    # ---------------------------------------------------------------
    # Step 3: BLS detection — search for transits with extended range
    # ---------------------------------------------------------------
    print("[3/5] Running BLS transit search (period range: 0.5–45 days)...")
    lc_for_bls = processed.to_lightcurve()

    # We search a wider period range to try to catch TOI-700 d (~37 days)
    candidate = run_bls_lightkurve(
        lc_for_bls,
        tic_id=TARGET_TIC,
        min_period=0.5,
        max_period=45.0,
        num_periods=20_000,
    )

    catalog = CandidateCatalog()

    if candidate is not None:
        print(f"      Found candidate: P={candidate.period:.4f} d, "
              f"depth={candidate.depth * 100:.3f}%, SNR={candidate.snr:.1f}")

        # Step 4: Validate
        print("[4/5] Validating candidate...")
        validation = validate_candidate(
            candidate,
            time=processed.time,
            flux=processed.flux,
        )
        catalog.add(candidate, validation)

        status = "VALID" if validation.is_valid else "REJECTED"
        print(f"      Status: {status}")
        if validation.flags:
            for flag in validation.flags:
                print(f"        - {flag}")

        # Step 5: Cross-match
        print("[5/5] Cross-matching with known catalogs...")
        xmatch = crossmatch_candidate(candidate)
        if xmatch.match_found:
            print(f"      MATCH: {xmatch.catalog_name} (ΔP={xmatch.period_difference:.4f} d)")
        else:
            print(f"      No match in known catalogs — potential new candidate!")
    else:
        print("      No transit signal detected.")
        print("[4/5] Skipping validation (no candidate)")
        print("[5/5] Skipping cross-match (no candidate)")

    # ---------------------------------------------------------------
    # Generate output plots
    # ---------------------------------------------------------------
    print("\nGenerating output plots...")
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_lightcurve_plot(processed, candidate, output_dir)
    if candidate is not None:
        _save_phase_plot(processed, candidate, output_dir)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(catalog.summary())

    # Export results
    csv_path = export_to_csv(catalog)
    print(f"\nResults saved to: {csv_path}")

    # Check if we found TOI-700 d (the most exciting planet — in the HZ!)
    if candidate is not None:
        _check_toi700d(candidate)

    print(f"\nOutput plots saved to: {output_dir}/")
    print(f"{'='*60}\n")


def _save_lightcurve_plot(processed, candidate, output_dir: Path) -> None:
    """Save a static light curve plot."""
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=processed.time,
        y=processed.flux,
        mode="markers",
        marker=dict(size=1.5, color="deepskyblue"),
        name="Processed flux",
    ))

    if candidate is not None:
        model_flux = transit_model_from_candidate(processed.time, candidate)
        fig.add_trace(go.Scatter(
            x=processed.time,
            y=model_flux,
            mode="lines",
            line=dict(color="red", width=1),
            name=f"Model (P={candidate.period:.3f} d)",
        ))

    fig.update_layout(
        title=f"{TARGET_NAME} ({TARGET_TIC}) — Light Curve",
        xaxis_title="Time (BTJD)",
        yaxis_title="Normalized Flux",
        template="plotly_dark",
        height=500,
        width=1200,
    )

    path = output_dir / "lightcurve.html"
    fig.write_html(str(path))
    logger.info("Saved light curve plot to %s", path)


def _save_phase_plot(processed, candidate, output_dir: Path) -> None:
    """Save a static phase-folded plot."""
    phase, flux_folded = phase_fold(
        processed.time, processed.flux, candidate.period, candidate.epoch
    )
    bin_centers, bin_means, bin_stds = bin_phase_curve(phase, flux_folded)

    model_phase = np.linspace(-0.5, 0.5, 1000)
    model_time = model_phase * candidate.period + candidate.epoch
    model_flux = transit_model_from_candidate(model_time, candidate)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=phase, y=flux_folded,
        mode="markers",
        marker=dict(size=1.5, color="rgba(100,149,237,0.2)"),
        name="Phase-folded",
    ))

    fig.add_trace(go.Scatter(
        x=bin_centers, y=bin_means,
        mode="markers",
        marker=dict(size=5, color="deepskyblue"),
        name="Binned",
    ))

    fig.add_trace(go.Scatter(
        x=model_phase, y=model_flux,
        mode="lines",
        line=dict(color="red", width=2),
        name="Transit model",
    ))

    fig.update_layout(
        title=(f"{TARGET_NAME} — Phase-Folded (P={candidate.period:.4f} d, "
               f"depth={candidate.depth * 100:.3f}%)"),
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        template="plotly_dark",
        height=500,
        width=1000,
    )

    path = output_dir / "phase_folded.html"
    fig.write_html(str(path))
    logger.info("Saved phase-folded plot to %s", path)


def _check_toi700d(candidate) -> None:
    """Check if we found TOI-700 d (period ~37.42 days)."""
    toi700d_period = 37.426

    if abs(candidate.period - toi700d_period) < 1.0:
        print("\n" + "*" * 60)
        print("  SUCCESS! Found TOI-700 d — a rocky planet in the")
        print("  habitable zone of an M-dwarf star!")
        print(f"  Detected period: {candidate.period:.4f} days")
        print(f"  Known period:    {toi700d_period:.4f} days")
        print("*" * 60)
    else:
        # Check other known planets
        known = {"TOI-700 b": 9.977, "TOI-700 c": 16.051, "TOI-700 d": 37.426}
        for name, period in known.items():
            if abs(candidate.period - period) < 0.5:
                print(f"\n  Found {name} (P={period} d) instead of TOI-700 d.")
                print(f"  Try extending the period search range to find planet d.")
                return
        print(f"\n  Detected period ({candidate.period:.4f} d) does not match")
        print(f"  any known TOI-700 planet. This could be a systematic artefact.")


if __name__ == "__main__":
    main()
