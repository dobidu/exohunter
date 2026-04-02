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


def make_sky_map(
    ra: np.ndarray,
    dec: np.ndarray,
    tic_ids: list[str],
    statuses: list[str],
    magnitudes: np.ndarray | None = None,
) -> go.Figure:
    """Create a sky map (RA/Dec scatter) of all processed targets.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        tic_ids: TIC IDs for hover labels.
        statuses: Status per target: ``"processed"``, ``"candidate"``,
            ``"validated"``, or ``"rejected"``.
        magnitudes: Optional TESS magnitudes for hover info.

    Returns:
        A Plotly ``Figure`` with colour-coded targets.
    """
    color_map = {
        "processed": "gray",
        "candidate": "gold",
        "validated": "limegreen",
        "rejected": "tomato",
    }

    fig = go.Figure()

    for status in ["processed", "candidate", "validated", "rejected"]:
        mask = [s == status for s in statuses]
        if not any(mask):
            continue

        indices = [i for i, m in enumerate(mask) if m]
        hover_text = [
            f"{tic_ids[i]}<br>RA={ra[i]:.4f}° Dec={dec[i]:.4f}°"
            + (f"<br>Tmag={magnitudes[i]:.1f}" if magnitudes is not None else "")
            for i in indices
        ]

        fig.add_trace(go.Scattergl(
            x=[ra[i] for i in indices],
            y=[dec[i] for i in indices],
            mode="markers",
            marker=dict(
                size=6,
                color=color_map[status],
                opacity=0.8,
            ),
            name=status.capitalize(),
            text=hover_text,
            hoverinfo="text",
            customdata=[tic_ids[i] for i in indices],
        ))

    fig.update_layout(
        title="Sky Map — Processed Targets",
        xaxis_title="Right Ascension (°)",
        yaxis_title="Declination (°)",
        xaxis=dict(autorange="reversed"),  # RA increases right-to-left
        height=500,
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

        # Highlight transit windows with vertical bands
        t_start = time[0]
        t_end = time[-1]
        transit_times = np.arange(
            candidate.epoch, t_end, candidate.period
        )
        for t0 in transit_times:
            if t_start <= t0 <= t_end:
                fig.add_vrect(
                    x0=t0 - candidate.duration / 2,
                    x1=t0 + candidate.duration / 2,
                    fillcolor="rgba(255, 0, 0, 0.08)",
                    line_width=0,
                    layer="below",
                )

    fig.update_layout(
        title="Light Curve",
        xaxis_title="Time (BTJD)",
        yaxis_title="Normalized Flux",
        height=400,
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
        marker=dict(size=1.5, color="rgba(100,149,237,0.2)"),
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
            f"SNR={candidate.snr:.1f}"
        ),
        xaxis_title="Orbital Phase",
        yaxis_title="Normalized Flux",
        height=400,
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
