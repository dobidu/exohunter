#!/usr/bin/env python3
"""Start the ExoHunter interactive dashboard.

Usage::

    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8080
    python scripts/run_dashboard.py --demo   # Load demo data (synthetic)
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


def generate_demo_data() -> dict:
    """Generate synthetic demo data for the dashboard.

    Creates a small set of fake targets and candidates so the dashboard
    can be explored without downloading real TESS data.

    Returns:
        A dictionary in the format expected by ``pipeline-data`` Store.
    """
    rng = np.random.default_rng(42)

    # Synthetic targets scattered across the sky
    n_targets = 50
    ra = rng.uniform(0, 360, n_targets)
    dec = rng.uniform(-90, 90, n_targets)
    tic_ids = [f"TIC {100000000 + i}" for i in range(n_targets)]

    # A few targets will have "detected" candidates
    candidate_indices = [0, 5, 12, 23, 37]
    statuses = ["processed"] * n_targets
    for idx in candidate_indices:
        statuses[idx] = "validated"

    targets = [
        {"tic_id": tic_ids[i], "ra": float(ra[i]), "dec": float(dec[i]),
         "status": statuses[i]}
        for i in range(n_targets)
    ]

    # Generate synthetic light curves and candidates for the "detected" targets
    candidates = []
    lightcurves = {}

    demo_planets = [
        {"tic_id": tic_ids[0], "period": 10.0, "depth": 0.01, "snr": 15.2},
        {"tic_id": tic_ids[5], "period": 3.5, "depth": 0.005, "snr": 9.8},
        {"tic_id": tic_ids[12], "period": 7.2, "depth": 0.008, "snr": 12.1},
        {"tic_id": tic_ids[23], "period": 1.8, "depth": 0.015, "snr": 22.5},
        {"tic_id": tic_ids[37], "period": 15.0, "depth": 0.003, "snr": 7.5},
    ]

    for planet in demo_planets:
        tic = planet["tic_id"]
        period = planet["period"]
        depth = planet["depth"]

        # Generate synthetic light curve with transit
        n_points = 15000
        time = np.linspace(0, 90, n_points)
        flux = np.ones(n_points)
        phase = (time % period) / period
        in_transit = np.abs(phase - 0.5) < 0.01
        flux[in_transit] -= depth
        flux += rng.normal(0, 0.001, n_points)

        lightcurves[tic] = {
            "time": time.tolist(),
            "flux": flux.tolist(),
        }

        candidates.append({
            "tic_id": tic,
            "period": period,
            "epoch": period * 0.5,
            "duration": period * 0.02,
            "depth": depth,
            "snr": planet["snr"],
            "bls_power": planet["snr"] * 0.1,
            "n_transits": int(90 / period),
            "status": "Validated",
            "flags": "",
        })

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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Load synthetic demo data instead of real pipeline results",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable Dash debug mode",
    )
    return parser.parse_args()


def main() -> None:
    """Start the dashboard server."""
    args = parse_args()

    pipeline_data = None
    if args.demo:
        logger.info("Loading synthetic demo data")
        pipeline_data = generate_demo_data()

    app = create_app(pipeline_data=pipeline_data)

    logger.info("Starting dashboard at http://%s:%d", args.host, args.port)
    app.run(
        host=args.host,
        port=args.port,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
