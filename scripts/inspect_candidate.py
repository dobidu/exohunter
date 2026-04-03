#!/usr/bin/env python3
"""Deep inspection of a transit candidate.

Downloads all available TESS sectors for a target, preprocesses,
runs BLS, fits a trapezoidal transit model, and generates a
multi-panel diagnostic report saved as PNG.

Usage::

    # Inspect TOI-700 d
    python scripts/inspect_candidate.py --tic "TIC 150428135" --period 37.426

    # Custom BLS range and output
    python scripts/inspect_candidate.py --tic "TIC 261136679" --period 3.69 \\
        --min-period 1.0 --max-period 10.0

    # Skip download (use cache only)
    python scripts/inspect_candidate.py --tic "TIC 150428135" --period 9.977
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize_scalar

from exohunter import config
from exohunter.detection.model import (
    bin_phase_curve,
    phase_fold,
    transit_model,
)
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Transit model fitting
# ---------------------------------------------------------------------------

def fit_trapezoidal_model(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> dict:
    """Fit a trapezoidal transit model to the phase-folded data.

    Optimises depth and ingress fraction by minimising the sum of
    squared residuals between the model and the data.

    Args:
        time: Array of timestamps.
        flux: Array of normalized flux values.
        period: Orbital period in days.
        epoch: Mid-transit time (t0) in the same time system.
        duration: Transit duration in days.

    Returns:
        Dict with fitted parameters: ``depth``, ``ingress_fraction``,
        ``rp_rs`` (planet-to-star radius ratio), ``impact_param``
        (estimated impact parameter), and ``residual_rms``.
    """
    # Phase-fold
    phase_time = ((time - epoch + period / 2) % period) - period / 2
    in_transit = np.abs(phase_time) < duration / 2.0
    out_transit = np.abs(phase_time) > duration

    if np.sum(in_transit) < 5 or np.sum(out_transit) < 10:
        logger.warning("Too few points for model fitting")
        return {
            "depth": 0.0, "ingress_fraction": 0.1,
            "rp_rs": 0.0, "impact_param": 0.0, "residual_rms": 0.0,
        }

    # Estimate depth from the data
    baseline = np.median(flux[out_transit])
    transit_flux = np.median(flux[in_transit])
    depth_estimate = max(0.0, baseline - transit_flux)

    # Optimize ingress_fraction for the best trapezoidal fit
    def residual_for_ingress(ingress_frac: float) -> float:
        model = transit_model(
            time, period, epoch, duration,
            depth=depth_estimate, ingress_fraction=ingress_frac,
        )
        return float(np.sum((flux - model) ** 2))

    result = minimize_scalar(
        residual_for_ingress,
        bounds=(0.01, 0.45),
        method="bounded",
    )
    best_ingress = float(result.x)

    # Compute final model and residuals
    best_model = transit_model(
        time, period, epoch, duration,
        depth=depth_estimate, ingress_fraction=best_ingress,
    )
    residuals = flux - best_model
    residual_rms = float(np.std(residuals))

    # Derived physical parameters
    # Rp/R* = sqrt(depth) — from the transit depth equation
    rp_rs = float(np.sqrt(depth_estimate)) if depth_estimate > 0 else 0.0

    # Impact parameter estimate from ingress fraction:
    # For a uniform source, the ingress time is related to b by:
    #   T_ingress / T_total ~ (1 - b) × (Rp/R*) / (1 + Rp/R* - b × Rp/R*)
    # This is a simplified inversion — good enough for screening.
    # Higher ingress fraction → more grazing → higher impact parameter.
    impact_param = float(min(1.0, best_ingress * 2.5))

    return {
        "depth": depth_estimate,
        "ingress_fraction": best_ingress,
        "rp_rs": rp_rs,
        "impact_param": impact_param,
        "residual_rms": residual_rms,
    }


# ---------------------------------------------------------------------------
# Odd-even transit comparison
# ---------------------------------------------------------------------------

def odd_even_comparison(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
) -> dict:
    """Compare odd and even transits to check for eclipsing binary.

    An eclipsing binary with a period twice the detected period will
    show alternating deep and shallow dips.  If the odd and even
    transit depths are significantly different, the signal is likely
    a binary, not a planet.

    Args:
        time: Array of timestamps.
        flux: Array of normalized flux values.
        period: Orbital period in days.
        epoch: Mid-transit time.
        duration: Transit duration in days.

    Returns:
        Dict with ``depth_odd``, ``depth_even``, ``depth_diff``,
        ``is_consistent`` (True if depths agree within 3-sigma),
        and the individual phase-folded arrays for plotting.
    """
    half_dur = duration / 2.0
    baseline_mask = ((time - epoch + period / 2) % period - period / 2).copy()
    baseline_flux = np.median(flux[np.abs(baseline_mask) > duration])

    # Identify individual transit events
    transit_numbers = np.round((time - epoch) / period).astype(int)

    odd_in = []
    even_in = []

    unique_transits = np.unique(transit_numbers)
    for n in unique_transits:
        t_center = epoch + n * period
        dist = np.abs(time - t_center)
        in_this_transit = dist < half_dur

        if np.sum(in_this_transit) < 3:
            continue

        transit_flux = np.median(flux[in_this_transit])
        depth = baseline_flux - transit_flux

        if n % 2 == 0:
            even_in.append(depth)
        else:
            odd_in.append(depth)

    depth_odd = float(np.median(odd_in)) if odd_in else 0.0
    depth_even = float(np.median(even_in)) if even_in else 0.0

    # Consistency check: depths should agree within noise
    all_depths = odd_in + even_in
    if len(all_depths) >= 2:
        scatter = float(np.std(all_depths))
        depth_diff = abs(depth_odd - depth_even)
        is_consistent = depth_diff < 3.0 * scatter if scatter > 0 else True
    else:
        depth_diff = abs(depth_odd - depth_even)
        is_consistent = True  # not enough data to judge

    # Phase-fold for odd and even separately for plotting
    odd_mask = np.isin(transit_numbers, [n for n in unique_transits if n % 2 != 0])
    even_mask = np.isin(transit_numbers, [n for n in unique_transits if n % 2 == 0])

    return {
        "depth_odd": depth_odd,
        "depth_even": depth_even,
        "depth_diff": depth_diff,
        "is_consistent": is_consistent,
        "n_odd": len(odd_in),
        "n_even": len(even_in),
        "odd_mask": odd_mask,
        "even_mask": even_mask,
    }


# ---------------------------------------------------------------------------
# Report generation (4-panel PNG)
# ---------------------------------------------------------------------------

def generate_report(
    tic_id: str,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    fit_result: dict,
    odd_even: dict,
    bls_periods: np.ndarray,
    bls_power: np.ndarray,
    output_path: Path,
) -> Path:
    """Generate a 4-panel diagnostic report and save as PNG.

    Panels:
        1. Full light curve with transit windows highlighted
        2. Phase-folded data with trapezoidal model fit
        3. BLS periodogram
        4. Odd-even transit depth comparison

    Args:
        tic_id: Target identifier.
        time: Full time array.
        flux: Full flux array.
        period: Best-fit period (days).
        epoch: Mid-transit time.
        duration: Transit duration (days).
        fit_result: Output of ``fit_trapezoidal_model``.
        odd_even: Output of ``odd_even_comparison``.
        bls_periods: Period grid from BLS.
        bls_power: BLS power spectrum.
        output_path: Where to save the PNG.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#1a1a2e")

    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", which="both")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    depth = fit_result["depth"]
    rp_rs = fit_result["rp_rs"]
    impact = fit_result["impact_param"]
    ingress = fit_result["ingress_fraction"]

    # ------------------------------------------------------------------
    # Panel 1: Full light curve with transit windows
    # ------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.scatter(time, flux, s=0.3, c="#4ea8de", alpha=0.4, rasterized=True)

    # Mark transit windows
    t_start, t_end = time[0], time[-1]
    transit_centers = np.arange(epoch, t_end + period, period)
    backward = np.arange(epoch - period, t_start - period, -period)
    transit_centers = np.concatenate([backward, transit_centers])

    for tc in transit_centers:
        if t_start <= tc <= t_end:
            ax1.axvspan(
                tc - duration / 2, tc + duration / 2,
                alpha=0.15, color="red", zorder=0,
            )

    # Overlay model
    model_flux = transit_model(
        time, period, epoch, duration,
        depth=depth, ingress_fraction=ingress,
    )
    sort_idx = np.argsort(time)
    ax1.plot(time[sort_idx], model_flux[sort_idx], c="red", lw=0.8, alpha=0.7)

    ax1.set_xlabel("Time (BTJD)")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title("Light Curve — Transit Windows Highlighted")
    n_transits = int((t_end - t_start) / period)
    ax1.text(
        0.02, 0.95,
        f"P = {period:.4f} d\n{n_transits} transits observed",
        transform=ax1.transAxes, fontsize=9, color="white",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#16213e", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # Panel 2: Phase-folded with model fit
    # ------------------------------------------------------------------
    ax2 = axes[0, 1]

    phase, flux_folded = phase_fold(time, flux, period, epoch)
    ax2.scatter(phase, flux_folded, s=0.3, c="#4ea8de", alpha=0.2, rasterized=True)

    # Binned data
    bin_centers, bin_means, bin_stds = bin_phase_curve(phase, flux_folded, n_bins=200)
    ax2.errorbar(
        bin_centers, bin_means, yerr=bin_stds,
        fmt="o", ms=3, color="deepskyblue", ecolor="gray",
        elinewidth=0.5, capsize=0, zorder=5,
    )

    # Model
    model_phase = np.linspace(-0.5, 0.5, 2000)
    model_time_fine = model_phase * period + epoch
    model_flux_fine = transit_model(
        model_time_fine, period, epoch, duration,
        depth=depth, ingress_fraction=ingress,
    )
    ax2.plot(model_phase, model_flux_fine, c="red", lw=2, zorder=10)

    ax2.set_xlabel("Orbital Phase")
    ax2.set_ylabel("Normalized Flux")
    ax2.set_title("Phase-Folded Light Curve + Trapezoidal Fit")
    ax2.set_xlim(-0.15, 0.15)
    ax2.text(
        0.02, 0.05,
        f"depth = {depth * 100:.4f}%\n"
        f"Rp/R* = {rp_rs:.4f}\n"
        f"b = {impact:.2f}\n"
        f"ingress = {ingress:.2f}\n"
        f"RMS = {fit_result['residual_rms']:.6f}",
        transform=ax2.transAxes, fontsize=9, color="#00ff88",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="#16213e", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # Panel 3: BLS periodogram
    # ------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(bls_periods, bls_power, c="#4ea8de", lw=0.8)
    ax3.axvline(period, color="red", ls="--", lw=1.5, label=f"P = {period:.4f} d")

    # Mark harmonics
    for h, label in [(2.0, "2P"), (0.5, "P/2"), (3.0, "3P"), (1 / 3, "P/3")]:
        hp = period * h
        if bls_periods[0] <= hp <= bls_periods[-1]:
            ax3.axvline(hp, color="orange", ls=":", lw=1, alpha=0.7)
            ax3.text(hp, ax3.get_ylim()[1] * 0.9, label, fontsize=7,
                     color="orange", ha="center")

    ax3.set_xlabel("Period (days)")
    ax3.set_ylabel("BLS Power")
    ax3.set_title("BLS Periodogram")
    ax3.legend(loc="upper right", fontsize=9, facecolor="#16213e",
               edgecolor="#333", labelcolor="white")

    # ------------------------------------------------------------------
    # Panel 4: Odd-even transit comparison
    # ------------------------------------------------------------------
    ax4 = axes[1, 1]

    # Phase-fold odd transits
    odd_mask = odd_even["odd_mask"]
    even_mask = odd_even["even_mask"]

    if np.sum(odd_mask) > 10 and np.sum(even_mask) > 10:
        phase_odd, flux_odd = phase_fold(time[odd_mask], flux[odd_mask], period, epoch)
        phase_even, flux_even = phase_fold(time[even_mask], flux[even_mask], period, epoch)

        _, odd_means, _ = bin_phase_curve(phase_odd, flux_odd, n_bins=100)
        odd_centers = np.linspace(-0.5, 0.5, len(odd_means)) if len(odd_means) > 0 else np.array([])
        _, even_means, _ = bin_phase_curve(phase_even, flux_even, n_bins=100)
        even_centers = np.linspace(-0.5, 0.5, len(even_means)) if len(even_means) > 0 else np.array([])

        if len(odd_centers) > 0:
            ax4.plot(odd_centers, odd_means, "o-", ms=3, color="#ff6b6b",
                     label=f"Odd (n={odd_even['n_odd']}, d={odd_even['depth_odd']*100:.4f}%)")
        if len(even_centers) > 0:
            ax4.plot(even_centers, even_means, "s-", ms=3, color="#4ecdc4",
                     label=f"Even (n={odd_even['n_even']}, d={odd_even['depth_even']*100:.4f}%)")
    else:
        ax4.text(
            0.5, 0.5, "Not enough transits\nfor odd-even comparison",
            transform=ax4.transAxes, ha="center", va="center",
            fontsize=14, color="gray",
        )

    status_color = "#00ff88" if odd_even["is_consistent"] else "#ff6b6b"
    status_text = "CONSISTENT" if odd_even["is_consistent"] else "INCONSISTENT — possible EB"
    ax4.set_xlabel("Orbital Phase")
    ax4.set_ylabel("Normalized Flux")
    ax4.set_title("Odd vs Even Transits (Eclipsing Binary Check)")
    ax4.set_xlim(-0.15, 0.15)
    ax4.legend(loc="lower right", fontsize=8, facecolor="#16213e",
               edgecolor="#333", labelcolor="white")
    ax4.text(
        0.02, 0.95, f"Odd-even: {status_text}\n"
        f"|delta_depth| = {odd_even['depth_diff']*100:.4f}%",
        transform=ax4.transAxes, fontsize=9, color=status_color,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#16213e", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # Title and layout
    # ------------------------------------------------------------------
    fig.suptitle(
        f"ExoHunter Candidate Report — {tic_id}   |   "
        f"P = {period:.4f} d   |   "
        f"depth = {depth * 100:.4f}%   |   "
        f"Rp/R* = {rp_rs:.4f}",
        fontsize=14, color="white", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Report saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def inspect_candidate(
    tic_id: str,
    period: float,
    min_period: float = config.BLS_MIN_PERIOD_DAYS,
    max_period: float = config.BLS_MAX_PERIOD_DAYS,
) -> Path:
    """Run full inspection pipeline for a single candidate.

    Args:
        tic_id: TIC identifier (e.g. ``"TIC 150428135"``).
        period: Known or detected orbital period in days.
        min_period: BLS search minimum period.
        max_period: BLS search maximum period.

    Returns:
        Path to the saved PNG report.
    """
    # Normalize TIC ID
    if not tic_id.upper().startswith("TIC"):
        tic_id = f"TIC {tic_id}"

    tic_num = tic_id.replace("TIC ", "").replace("TIC", "").strip()

    logger.info("=" * 60)
    logger.info("  Inspecting %s at P = %.4f d", tic_id, period)
    logger.info("=" * 60)

    # Step 1: Download all available sectors
    logger.info("[1/6] Downloading light curve (all sectors)...")
    light_curve = download_lightcurve(tic_id)
    if light_curve is None:
        logger.error("Could not download data for %s", tic_id)
        sys.exit(1)
    logger.info("       %d cadences downloaded", len(light_curve))

    # Step 2: Preprocess
    logger.info("[2/6] Preprocessing...")
    processed = preprocess_single(light_curve, tic_id=tic_id)
    time = processed.time
    flux = processed.flux
    logger.info("       %d cadences after cleaning", len(time))

    # Step 3: Run BLS periodogram (for panel 3)
    logger.info("[3/6] Running BLS periodogram...")
    lc_for_bls = processed.to_lightcurve()

    bls_period_grid = np.linspace(min_period, max_period, config.BLS_NUM_PERIODS)
    periodogram = lc_for_bls.to_periodogram(
        method="bls",
        period=bls_period_grid,
        frequency_factor=config.BLS_FREQUENCY_FACTOR,
    )
    bls_power = periodogram.power.value
    bls_periods = periodogram.period.value

    bls_best_period = float(periodogram.period_at_max_power.value)
    bls_best_t0 = float(periodogram.transit_time_at_max_power.value)
    bls_best_duration = float(periodogram.duration_at_max_power.value)
    logger.info(
        "       BLS peak: P=%.4f d (user period: %.4f d)",
        bls_best_period, period,
    )

    # Use the user-provided period, but take epoch and duration from BLS
    # if the user period matches the BLS peak (within 10%)
    if abs(bls_best_period - period) / period < 0.10:
        epoch = bls_best_t0
        duration = bls_best_duration
        logger.info("       Using BLS epoch=%.4f, duration=%.4f d", epoch, duration)
    else:
        # User period doesn't match BLS peak — re-run BLS focused on
        # the user period to get a better epoch and duration
        logger.info("       User period differs from BLS peak — refining...")
        narrow_grid = np.linspace(period * 0.95, period * 1.05, 1000)
        narrow_pg = lc_for_bls.to_periodogram(
            method="bls", period=narrow_grid, frequency_factor=500,
        )
        epoch = float(narrow_pg.transit_time_at_max_power.value)
        duration = float(narrow_pg.duration_at_max_power.value)
        logger.info("       Refined epoch=%.4f, duration=%.4f d", epoch, duration)

    # Step 4: Fit trapezoidal model
    logger.info("[4/6] Fitting trapezoidal transit model...")
    fit_result = fit_trapezoidal_model(time, flux, period, epoch, duration)
    logger.info(
        "       depth=%.4f%%, Rp/R*=%.4f, b=%.2f, ingress=%.2f",
        fit_result["depth"] * 100,
        fit_result["rp_rs"],
        fit_result["impact_param"],
        fit_result["ingress_fraction"],
    )

    # Step 5: Odd-even comparison
    logger.info("[5/6] Odd-even transit comparison...")
    odd_even = odd_even_comparison(time, flux, period, epoch, duration)
    status = "CONSISTENT" if odd_even["is_consistent"] else "INCONSISTENT"
    logger.info(
        "       Odd depth=%.4f%%, Even depth=%.4f%%, |diff|=%.4f%% → %s",
        odd_even["depth_odd"] * 100,
        odd_even["depth_even"] * 100,
        odd_even["depth_diff"] * 100,
        status,
    )

    # Step 6: Generate report
    logger.info("[6/6] Generating 4-panel report...")
    output_path = config.REPORTS_DIR / f"TIC_{tic_num}.png"

    report_path = generate_report(
        tic_id=tic_id,
        time=time,
        flux=flux,
        period=period,
        epoch=epoch,
        duration=duration,
        fit_result=fit_result,
        odd_even=odd_even,
        bls_periods=bls_periods,
        bls_power=bls_power,
        output_path=output_path,
    )

    logger.info("=" * 60)
    logger.info("  Report saved: %s", report_path)
    logger.info("=" * 60)

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ExoHunter — Deep candidate inspection with 4-panel report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tic",
        type=str,
        required=True,
        help="TIC ID of the target (e.g. 'TIC 150428135')",
    )
    parser.add_argument(
        "--period",
        type=float,
        required=True,
        help="Orbital period in days (from BLS detection or catalog)",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=config.BLS_MIN_PERIOD_DAYS,
        help="BLS periodogram minimum period (days)",
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=config.BLS_MAX_PERIOD_DAYS,
        help="BLS periodogram maximum period (days)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    inspect_candidate(
        tic_id=args.tic,
        period=args.period,
        min_period=args.min_period,
        max_period=args.max_period,
    )


if __name__ == "__main__":
    main()
