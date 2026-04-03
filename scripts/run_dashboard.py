#!/usr/bin/env python3
"""Start the ExoHunter interactive dashboard.

Usage::

    python scripts/run_dashboard.py             # default: loads demo (TOI-700)
    python scripts/run_dashboard.py --demo      # explicit demo mode
    python scripts/run_dashboard.py --csv       # load from data/output/candidates.csv
    python scripts/run_dashboard.py --empty     # start with empty dashboard
    python scripts/run_dashboard.py --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from exohunter import config
from exohunter.dashboard.app import create_app
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# TOI-700 system — real astrophysical parameters
# ---------------------------------------------------------------------------
# TOI-700 (TIC 150428135) is an M2-dwarf at ~31 pc in Dorado.
# TESS observed it in multiple southern sectors.
#
# Planet parameters from Gilbert et al. (2023) and Kostov et al. (2020):
#   b — P = 9.977 d,  Rp ≈ 1.01 R⊕, depth ≈ 580 ppm
#   c — P = 16.051 d, Rp ≈ 2.63 R⊕, depth ≈ 780 ppm
#   d — P = 37.426 d, Rp ≈ 1.14 R⊕, depth ≈ 810 ppm  (habitable zone!)
# ---------------------------------------------------------------------------

TOI700_TIC = "TIC 150428135"
TOI700_RA = 97.2049      # degrees (J2000)
TOI700_DEC = -65.5768     # degrees (J2000)
TOI700_TMAG = 13.15       # TESS magnitude

TOI700_PLANETS = [
    {
        "name": "TOI-700 b",
        "period": 9.977,
        "epoch": 2.5,       # arbitrary offset within the synthetic window
        "depth": 0.00058,   # 580 ppm
        "duration": 0.095,  # ~2.3 hours
        "snr": 12.4,
    },
    {
        "name": "TOI-700 c",
        "period": 16.051,
        "epoch": 4.1,
        "depth": 0.00078,   # 780 ppm
        "duration": 0.125,  # ~3.0 hours
        "snr": 10.8,
    },
    {
        "name": "TOI-700 d",
        "period": 37.426,
        "epoch": 8.7,
        "depth": 0.00081,   # 810 ppm — habitable zone planet!
        "duration": 0.148,  # ~3.5 hours
        "snr": 8.1,
    },
]


def _generate_toi700_lightcurve(
    baseline_days: float = 351.0,
    n_points: int = 25_000,
    noise_ppm: float = 300.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic TESS light curve for TOI-700 with all 3 planets.

    The light curve mimics ~13 TESS sectors of observation (351 days) at
    2-minute cadence with realistic photometric noise for an M-dwarf
    (CDPP ~300 ppm).

    Args:
        baseline_days: Total observation span in days.
        n_points: Number of cadences.
        noise_ppm: Gaussian noise standard deviation in parts per million.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(time, flux)`` numpy arrays.
    """
    rng = np.random.default_rng(seed)
    noise_frac = noise_ppm * 1e-6

    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones(n_points, dtype=np.float64)

    # Inject box-shaped transits for each planet
    for planet in TOI700_PLANETS:
        period = planet["period"]
        epoch = planet["epoch"]
        depth = planet["depth"]
        duration = planet["duration"]
        half_dur = duration / 2.0

        # Phase-fold to find in-transit cadences
        phase_time = ((time - epoch + period / 2) % period) - period / 2
        in_transit = np.abs(phase_time) < half_dur
        flux[in_transit] -= depth

    # Add realistic Gaussian noise
    flux += rng.normal(0, noise_frac, n_points)

    # Simulate slow stellar variability (~0.1% amplitude, ~10-day timescale)
    # This is typical for a quiet M-dwarf observed by TESS
    variability = 0.001 * np.sin(2 * np.pi * time / 11.3)
    variability += 0.0005 * np.sin(2 * np.pi * time / 5.7 + 1.2)
    flux += variability

    return time, flux


def _generate_sector_stars(
    n_stars: int = 40,
    seed: int = 123,
) -> list[dict]:
    """Generate simulated background stars from the same TESS sector.

    Places stars in a realistic field around TOI-700's sky coordinates,
    mimicking what a TESS sector observation looks like.

    Args:
        n_stars: Number of background stars to generate.
        seed: Random seed.

    Returns:
        List of target dicts with ``tic_id``, ``ra``, ``dec``, ``status``.
    """
    rng = np.random.default_rng(seed)

    # TESS has a 24°×96° field of view per sector — we simulate a small
    # patch around TOI-700's position with a spread of ~5° in each axis
    ra_offsets = rng.normal(0, 2.5, n_stars)
    dec_offsets = rng.normal(0, 2.5, n_stars)

    targets = []
    for i in range(n_stars):
        # Assign some stars as "candidate" or "rejected" for visual variety
        roll = rng.random()
        if roll < 0.10:
            status = "candidate"
        elif roll < 0.15:
            status = "rejected"
        else:
            status = "processed"

        targets.append({
            "tic_id": f"TIC {200000000 + i}",
            "ra": float(TOI700_RA + ra_offsets[i]),
            "dec": float(TOI700_DEC + dec_offsets[i]),
            "status": status,
            "tmag": float(rng.uniform(10.0, 16.0)),
        })

    return targets


def generate_demo_data() -> dict:
    """Generate realistic TOI-700 demonstration data for the dashboard.

    Produces a synthetic multi-planet light curve that closely mimics
    real TESS observations of the TOI-700 system, along with simulated
    background stars for the sky map.

    Returns:
        A dictionary in the ``pipeline-data`` Store schema.
    """
    logger.info("Generating TOI-700 demo data (3 planets, realistic noise)")

    # --- Light curve with all 3 planets superimposed ---
    time, flux = _generate_toi700_lightcurve()

    lightcurves = {
        TOI700_TIC: {
            "time": time.tolist(),
            "flux": flux.tolist(),
        },
    }

    # --- Candidates (one per planet) ---
    candidates = []
    for planet in TOI700_PLANETS:
        candidates.append({
            "tic_id": TOI700_TIC,
            "period": planet["period"],
            "epoch": planet["epoch"],
            "duration": planet["duration"],
            "depth": planet["depth"],
            "snr": planet["snr"],
            "bls_power": planet["snr"] * 0.08,
            "n_transits": max(1, int(351.0 / planet["period"])),
            "status": "validated",
            "flags": "",
            "name": planet["name"],
        })

    # --- Sky map targets: TOI-700 + simulated sector stars ---
    background_stars = _generate_sector_stars(n_stars=40)

    # TOI-700 itself — highlighted as validated
    toi700_target = {
        "tic_id": TOI700_TIC,
        "ra": TOI700_RA,
        "dec": TOI700_DEC,
        "status": "validated",
        "tmag": TOI700_TMAG,
    }
    targets = [toi700_target] + background_stars

    return {
        "targets": targets,
        "candidates": candidates,
        "lightcurves": lightcurves,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ExoHunter Dashboard — Interactive Transit Visualization",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.DASHBOARD_HOST,
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.DASHBOARD_PORT,
        help="Port to bind the server to",
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--demo",
        action="store_const",
        const="demo",
        dest="data_source",
        help="Load synthetic TOI-700 demo data (default)",
    )
    source_group.add_argument(
        "--csv",
        action="store_const",
        const="csv",
        dest="data_source",
        help="Load candidates from data/output/candidates.csv",
    )
    source_group.add_argument(
        "--empty",
        action="store_const",
        const="none",
        dest="data_source",
        help="Start with an empty dashboard",
    )
    parser.set_defaults(data_source=None)

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable Dash debug mode",
    )
    return parser.parse_args()


def main() -> None:
    """Start the dashboard server."""
    args = parse_args()

    # Determine data source: CLI flag > config file
    data_source = args.data_source or config.DASHBOARD_DATA_SOURCE

    pipeline_data = None
    if data_source == "demo":
        logger.info("Loading TOI-700 demo data")
        pipeline_data = generate_demo_data()
    elif data_source == "csv":
        logger.info("Loading pipeline results from CSV (not yet implemented)")
        # TODO: Students can implement CSV → pipeline-data bridge here
        pipeline_data = None
    else:
        logger.info("Starting with empty dashboard")

    app = create_app(pipeline_data=pipeline_data)

    logger.info("Starting dashboard at http://%s:%d", args.host, args.port)
    app.run(
        host=args.host,
        port=args.port,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
