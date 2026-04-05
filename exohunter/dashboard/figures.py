"""Plotly figure generation for the ExoHunter dashboard.

Each function returns a ``plotly.graph_objects.Figure`` that can be
rendered directly by a Dash ``dcc.Graph`` component.  All figures
use a consistent dark theme to match the DARKLY Bootstrap theme.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from exohunter.detection.bls import TransitCandidate
from exohunter.detection.model import (
    bin_phase_curve,
    phase_fold,
    transit_model_from_candidate,
)

# Consistent dark styling for all figures
DARK_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.1)"


def _apply_dark_style(fig: go.Figure) -> go.Figure:
    """Apply consistent dark styling to a figure."""
    fig.update_layout(
        template=DARK_TEMPLATE,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color="#ddd"),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def _milky_way_band() -> tuple[list[float], list[float]]:
    """Generate the Milky Way galactic plane as a band in RA/Dec.

    Returns a simplified polygon tracing the galactic plane
    (galactic latitude b=0) transformed to equatorial coordinates,
    with a +-15 deg width band.

    Returns:
        Tuple of (ra_list, dec_list) in degrees for a filled polygon.
    """
    # Galactic plane sampled at 10-deg intervals in galactic longitude,
    # transformed to equatorial J2000 using the standard rotation.
    # These are pre-computed points along b=0 (galactic equator).
    galactic_plane_ra = [
        266.4, 276.0, 286.5, 298.2, 311.5, 326.0, 340.4, 353.6,
        5.5, 16.5, 27.0, 37.3, 48.0, 59.3, 71.5, 84.5, 98.0,
        111.5, 124.0, 135.0, 145.0, 154.5, 164.0, 174.5, 186.5,
        200.0, 215.0, 230.5, 244.5, 255.0, 262.0, 266.4,
    ]
    galactic_plane_dec = [
        -28.9, -23.5, -15.5, -5.0, 6.5, 17.5, 26.0, 30.5,
        30.5, 27.0, 20.5, 12.5, 3.5, -5.5, -13.5, -19.5, -22.0,
        -20.0, -14.5, -6.5, 2.5, 11.5, 20.5, 29.0, 35.5,
        38.5, 36.0, 28.0, 17.0, 5.0, -10.0, -28.9,
    ]
    return galactic_plane_ra, galactic_plane_dec


def make_sky_map(
    ra: np.ndarray,
    dec: np.ndarray,
    tic_ids: list[str],
    statuses: list[str],
    magnitudes: np.ndarray | None = None,
    depths: np.ndarray | None = None,
) -> go.Figure:
    """Create an Aitoff-projected sky map with Milky Way band.

    Uses Plotly's ``Scattergeo`` with an ``aitoff`` projection for a
    realistic all-sky view. Markers are color-coded by status and
    sized by transit depth (deeper = larger). A semi-transparent
    Milky Way band provides spatial context.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        tic_ids: TIC IDs for hover labels.
        statuses: Status per target.
        magnitudes: Optional TESS magnitudes for hover info.
        depths: Optional transit depths (fractional) for marker sizing.

    Returns:
        A Plotly ``Figure`` with Aitoff-projected sky map.
    """
    color_map = {
        "processed": "rgba(150, 150, 150, 0.4)",
        "candidate": "gold",
        "validated": "limegreen",
        "new_candidate": "#00ff88",
        "known_match": "deepskyblue",
        "known_toi": "gold",
        "harmonic": "orange",
        "rejected": "tomato",
    }
    # Base sizes — will be scaled by depth if available
    base_size_map = {
        "processed": 4,
        "candidate": 7,
        "validated": 10,
        "new_candidate": 12,
        "known_match": 9,
        "known_toi": 8,
        "harmonic": 6,
        "rejected": 5,
    }

    # Convert RA from [0, 360] to longitude [-180, 180] for Aitoff
    lon = np.where(ra > 180, ra - 360, ra)

    fig = go.Figure()

    # --- Milky Way band (semi-transparent background) ---
    mw_ra, mw_dec = _milky_way_band()
    mw_lon = [r - 360 if r > 180 else r for r in mw_ra]
    fig.add_trace(go.Scattergeo(
        lon=mw_lon, lat=mw_dec,
        mode="lines",
        fill="toself",
        fillcolor="rgba(200, 180, 120, 0.08)",
        line=dict(color="rgba(200, 180, 120, 0.2)", width=1),
        name="Milky Way",
        hoverinfo="skip",
        showlegend=True,
    ))

    # --- Target markers, drawn in order so interesting ones are on top ---
    for status in ["processed", "rejected", "harmonic", "known_toi",
                    "candidate", "known_match", "validated", "new_candidate"]:
        mask = [s == status for s in statuses]
        if not any(mask):
            continue

        indices = [i for i, m in enumerate(mask) if m]

        # Compute marker sizes: base size + depth scaling
        base = base_size_map.get(status, 5)
        if depths is not None:
            sizes = []
            for i in indices:
                d = depths[i] if i < len(depths) else 0
                # Scale: depth of 0.01 (1%) adds +8 to base size
                depth_bonus = min(12, d * 800) if d > 0 else 0
                sizes.append(base + depth_bonus)
        else:
            sizes = [base] * len(indices)

        hover_text = []
        for i in indices:
            text = f"<b>{tic_ids[i]}</b><br>RA={ra[i]:.4f}°  Dec={dec[i]:.4f}°"
            if magnitudes is not None and i < len(magnitudes):
                text += f"<br>Tmag={magnitudes[i]:.1f}"
            if depths is not None and i < len(depths) and depths[i] > 0:
                text += f"<br>depth={depths[i]*100:.3f}%"
            hover_text.append(text)

        fig.add_trace(go.Scattergeo(
            lon=[float(lon[i]) for i in indices],
            lat=[float(dec[i]) for i in indices],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color_map.get(status, "gray"),
                opacity=0.9,
                line=dict(width=1, color="white") if status in ("validated", "new_candidate") else dict(width=0),
            ),
            name=status.replace("_", " ").capitalize(),
            text=hover_text,
            hoverinfo="text",
            customdata=[tic_ids[i] for i in indices],
        ))

    fig.update_geos(
        projection_type="aitoff",
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        bgcolor="rgba(0,0,0,0)",
        framecolor="rgba(100,100,100,0.3)",
        lonaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            dtick=30,
        ),
        lataxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            dtick=15,
        ),
    )

    fig.update_layout(
        title="Sky Map — Aitoff Projection",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return _apply_dark_style(fig)


def make_lightcurve_plot(
    time: np.ndarray,
    flux_raw: np.ndarray | None,
    flux_processed: np.ndarray,
    candidate: TransitCandidate | None = None,
    show_model: bool = True,
) -> go.Figure:
    """Create a light curve plot with optional transit model overlay.

    Args:
        time: Array of timestamps (BTJD).
        flux_raw: Raw flux array (before preprocessing). ``None`` to skip.
        flux_processed: Processed (detrended) flux array.
        candidate: Transit candidate — if provided, transit windows
            are highlighted and the model is overlaid.
        show_model: Whether to show the transit model.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure()

    # Raw light curve (semi-transparent background)
    if flux_raw is not None:
        fig.add_trace(go.Scattergl(
            x=time,
            y=flux_raw,
            mode="markers",
            marker=dict(size=1.5, color="rgba(100,100,100,0.3)"),
            name="Raw",
        ))

    # Processed light curve
    fig.add_trace(go.Scattergl(
        x=time,
        y=flux_processed,
        mode="markers",
        marker=dict(size=2, color="deepskyblue"),
        name="Processed",
    ))

    # Transit model overlay
    if candidate is not None and show_model:
        model_flux = transit_model_from_candidate(time, candidate)
        fig.add_trace(go.Scatter(
            x=time,
            y=model_flux,
            mode="lines",
            line=dict(color="red", width=1.5),
            name=f"Model (P={candidate.period:.3f} d)",
        ))

        # Highlight transit windows with vertical bands.
        # Generate transit times in BOTH directions from the epoch
        # so we cover the full observation span.
        t_start = float(time[0])
        t_end = float(time[-1])

        transit_times_forward = np.arange(candidate.epoch, t_end + candidate.period, candidate.period)
        transit_times_backward = np.arange(candidate.epoch - candidate.period, t_start - candidate.period, -candidate.period)
        all_transit_times = np.concatenate([transit_times_backward, transit_times_forward])

        for t0 in all_transit_times:
            if t_start <= t0 <= t_end:
                fig.add_vrect(
                    x0=t0 - candidate.duration / 2,
                    x1=t0 + candidate.duration / 2,
                    fillcolor="rgba(255, 0, 0, 0.08)",
                    line_width=0,
                    layer="below",
                )

    title = "Light Curve"
    if candidate is not None:
        display_name = candidate.name or candidate.tic_id
        title = f"Light Curve — {display_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Time (BTJD)",
        yaxis_title="Normalized Flux",
        height=450,
        xaxis_rangeslider_visible=True,
    )

    return _apply_dark_style(fig)


def make_phase_plot(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: TransitCandidate,
    n_bins: int = 200,
) -> go.Figure:
    """Create a phase-folded light curve plot.

    Args:
        time: Array of timestamps.
        flux: Array of processed flux values.
        candidate: The transit candidate (provides period and epoch).
        n_bins: Number of bins for the binned curve.

    Returns:
        A Plotly ``Figure`` with raw phase-folded data, binned data,
        and the transit model.
    """
    # Phase-fold the data
    phase, flux_folded = phase_fold(time, flux, candidate.period, candidate.epoch)

    # Bin the phase-folded data
    bin_centers, bin_means, bin_stds = bin_phase_curve(phase, flux_folded, n_bins=n_bins)

    # Generate model on a fine phase grid
    model_phase = np.linspace(-0.5, 0.5, 1000)
    model_time = model_phase * candidate.period + candidate.epoch
    model_flux = transit_model_from_candidate(model_time, candidate)

    fig = go.Figure()

    # Raw phase-folded data (semi-transparent)
    fig.add_trace(go.Scattergl(
        x=phase,
        y=flux_folded,
        mode="markers",
        marker=dict(size=1.5, color="rgba(100,149,237,0.15)"),
        name="Phase-folded data",
    ))

    # Binned data with error bars
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_means,
        error_y=dict(type="data", array=bin_stds, visible=True, thickness=1),
        mode="markers",
        marker=dict(size=5, color="deepskyblue"),
        name="Binned",
    ))

    # Transit model
    fig.add_trace(go.Scatter(
        x=model_phase,
        y=model_flux,
        mode="lines",
        line=dict(color="red", width=2),
        name="Transit model",
    ))

    fig.update_layout(
        title=(
            f"Phase-Folded Light Curve — "
            f"P={candidate.period:.4f} d, "
            f"depth={candidate.depth * 100:.3f}%, "
            f"duration={candidate.duration * 24:.1f} h, "
            f"SNR={candidate.snr:.1f}"
        ),
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        height=400,
    )

    return _apply_dark_style(fig)


def make_periodogram_plot(
    bls_periods: np.ndarray,
    bls_power: np.ndarray,
    candidate: TransitCandidate,
) -> go.Figure:
    """Create a BLS periodogram plot with peak and harmonics marked.

    Args:
        bls_periods: Array of trial periods (days).
        bls_power: Array of BLS power values.
        candidate: The detected candidate (provides the peak period).

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bls_periods,
        y=bls_power,
        mode="lines",
        line=dict(color="deepskyblue", width=1),
        name="BLS Power",
    ))

    # Mark the detected period
    fig.add_vline(
        x=candidate.period,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"P={candidate.period:.4f} d",
        annotation_font_color="red",
        annotation_position="top right",
    )

    # Mark harmonics
    harmonic_labels = [(2.0, "2P"), (0.5, "P/2"), (3.0, "3P"), (1 / 3, "P/3")]
    p_min, p_max = float(bls_periods[0]), float(bls_periods[-1])
    for ratio, label in harmonic_labels:
        hp = candidate.period * ratio
        if p_min <= hp <= p_max:
            fig.add_vline(
                x=hp,
                line=dict(color="orange", width=1, dash="dot"),
                annotation_text=label,
                annotation_font_color="orange",
                annotation_font_size=10,
                annotation_position="top left",
            )

    display_name = candidate.name or candidate.tic_id
    fig.update_layout(
        title=f"BLS Periodogram — {display_name}",
        xaxis_title="Period (days)",
        yaxis_title="BLS Power",
        height=350,
    )

    return _apply_dark_style(fig)


def make_odd_even_plot(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: TransitCandidate,
    n_bins: int = 100,
) -> go.Figure:
    """Create an odd-even transit comparison plot.

    Splits transits into odd-numbered and even-numbered events, phase-folds
    each subset, and overlays the binned curves. If the depths differ
    significantly, the signal may be an eclipsing binary at twice the
    detected period.

    Args:
        time: Array of timestamps.
        flux: Array of normalized flux values.
        candidate: The transit candidate.
        n_bins: Number of phase bins per subset.

    Returns:
        A Plotly ``Figure`` with odd and even binned phase curves.
    """
    period = candidate.period
    epoch = candidate.epoch
    duration = candidate.duration

    # Identify individual transit events by number
    transit_numbers = np.round((time - epoch) / period).astype(int)
    half_dur = duration / 2.0

    # Compute baseline flux
    phase_time = ((time - epoch + period / 2) % period) - period / 2
    out_mask = np.abs(phase_time) > duration
    baseline = float(np.median(flux[out_mask])) if np.sum(out_mask) > 10 else 1.0

    # Collect per-transit depths for odd and even
    unique_transits = np.unique(transit_numbers)
    odd_depths: list[float] = []
    even_depths: list[float] = []

    for n in unique_transits:
        t_center = epoch + n * period
        in_this = np.abs(time - t_center) < half_dur
        if np.sum(in_this) < 3:
            continue
        d = baseline - float(np.median(flux[in_this]))
        if n % 2 == 0:
            even_depths.append(d)
        else:
            odd_depths.append(d)

    depth_odd = float(np.median(odd_depths)) if odd_depths else 0.0
    depth_even = float(np.median(even_depths)) if even_depths else 0.0
    depth_diff = abs(depth_odd - depth_even)

    # Consistency check
    all_depths = odd_depths + even_depths
    if len(all_depths) >= 2:
        scatter = float(np.std(all_depths))
        is_consistent = depth_diff < 3.0 * scatter if scatter > 0 else True
    else:
        is_consistent = True

    # Phase-fold odd and even separately
    odd_mask = np.isin(transit_numbers, [n for n in unique_transits if n % 2 != 0])
    even_mask = np.isin(transit_numbers, [n for n in unique_transits if n % 2 == 0])

    fig = go.Figure()

    for mask, label, color, symbol, depths, n_count in [
        (odd_mask, "Odd transits", "#ff6b6b", "circle", odd_depths, len(odd_depths)),
        (even_mask, "Even transits", "#4ecdc4", "square", even_depths, len(even_depths)),
    ]:
        if np.sum(mask) < 10:
            continue
        phase_sub, flux_sub = phase_fold(time[mask], flux[mask], period, epoch)
        centers, means, stds = bin_phase_curve(phase_sub, flux_sub, n_bins=n_bins)

        if len(centers) > 0:
            depth_val = float(np.median(depths)) if depths else 0.0
            fig.add_trace(go.Scatter(
                x=centers,
                y=means,
                error_y=dict(type="data", array=stds, visible=True, thickness=0.8),
                mode="markers+lines",
                marker=dict(size=4, color=color, symbol=symbol),
                line=dict(color=color, width=1),
                name=f"{label} (n={n_count}, d={depth_val * 100:.4f}%)",
            ))

    status_text = "CONSISTENT — likely planet" if is_consistent else "INCONSISTENT — possible eclipsing binary"
    status_color = "#00ff88" if is_consistent else "#ff6b6b"

    display_name = candidate.name or candidate.tic_id
    fig.update_layout(
        title=(
            f"Odd vs Even Transits — {display_name}   "
            f"<span style='color:{status_color}'>{status_text}</span>"
        ),
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        xaxis_range=[-0.15, 0.15],
        height=350,
    )

    # Annotation with depth difference
    fig.add_annotation(
        text=(
            f"|Δdepth| = {depth_diff * 100:.4f}%<br>"
            f"Odd: {depth_odd * 100:.4f}%  Even: {depth_even * 100:.4f}%"
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.05,
        showarrow=False,
        font=dict(size=11, color=status_color),
        align="left",
        bgcolor="rgba(22,33,62,0.8)",
        bordercolor=status_color,
        borderwidth=1,
        borderpad=6,
    )

    return _apply_dark_style(fig)


def make_empty_figure(message: str = "Select a target to view") -> go.Figure:
    """Create an empty placeholder figure with a message.

    Args:
        message: Text to display in the centre of the figure.

    Returns:
        A Plotly ``Figure`` with the message as an annotation.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
    )
    return _apply_dark_style(fig)
