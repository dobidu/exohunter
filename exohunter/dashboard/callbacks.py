"""Dash callbacks for interactive behaviour.

Each callback connects user interactions (clicks, slider changes)
to updates in the dashboard visualizations.

Data flow:
    1. On page load, ``pipeline-data`` store is populated with demo data.
    2. Filters and pipeline-data together populate the candidate table.
    3. Clicking a row in the table (or a point on the sky map) selects
       a target and populates the candidate selector dropdown.
    4. The selected candidate updates the light curve and phase plots.
    5. The export button triggers a CSV download.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, no_update

from exohunter.dashboard.figures import (
    make_empty_figure,
    make_lightcurve_plot,
    make_phase_plot,
    make_sky_map,
)
from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def _candidate_from_dict(data: dict) -> TransitCandidate:
    """Reconstruct a ``TransitCandidate`` from a JSON-serializable dict.

    Centralises the dict-to-dataclass conversion that was previously
    duplicated across multiple callbacks.
    """
    return TransitCandidate(
        tic_id=data["tic_id"],
        period=data["period"],
        epoch=data["epoch"],
        duration=data["duration"],
        depth=data["depth"],
        snr=data["snr"],
        bls_power=data.get("bls_power", 0),
        n_transits=data.get("n_transits", 0),
        name=data.get("name", ""),
    )


def _candidates_for_target(
    tic_id: str,
    pipeline_data: dict[str, Any],
) -> list[dict]:
    """Return all candidate dicts matching a given TIC ID."""
    return [
        c for c in pipeline_data.get("candidates", [])
        if c.get("tic_id") == tic_id
    ]


def register_callbacks(app: Dash) -> None:
    """Register all Dash callbacks on the given app instance.

    Args:
        app: The Dash application.
    """

    # ------------------------------------------------------------------
    # Candidate table — now also triggered by pipeline-data to
    # auto-populate on page load (Bug #1 fix).
    # ------------------------------------------------------------------
    @app.callback(
        Output("candidate-table", "data"),
        Input("period-range", "value"),
        Input("min-snr", "value"),
        Input("status-filter", "value"),
        Input("pipeline-data", "data"),
    )
    def update_candidate_table(
        period_range: list[float],
        min_snr: float,
        status_filter: list[str],
        pipeline_data: dict[str, Any],
    ) -> list[dict]:
        """Filter and update the candidate table based on sidebar controls."""
        candidates = pipeline_data.get("candidates", [])
        if not candidates:
            return []

        filtered = []
        for c in candidates:
            period = c.get("period", 0)
            snr = c.get("snr", 0)
            status = c.get("status", "candidate").lower()

            if period < period_range[0] or period > period_range[1]:
                continue
            if snr < min_snr:
                continue
            if status not in status_filter:
                continue

            filtered.append({
                "tic_id": c.get("tic_id", ""),
                "name": c.get("name", ""),
                "period": c.get("period", 0),
                "epoch": c.get("epoch", 0),
                "duration": c.get("duration", 0),
                "depth_pct": c.get("depth", 0) * 100,
                "snr": c.get("snr", 0),
                "score": c.get("score", 0),
                "xmatch_class": c.get("xmatch_class", ""),
                "status": c.get("status", "Candidate").capitalize(),
                "flags": c.get("flags", ""),
            })

        return filtered

    # ------------------------------------------------------------------
    # Sky map — renders once when pipeline-data arrives.
    # ------------------------------------------------------------------
    @app.callback(
        Output("sky-map", "figure"),
        Input("pipeline-data", "data"),
    )
    def update_sky_map(pipeline_data: dict[str, Any]) -> Any:
        """Render the sky map from pipeline data."""
        targets = pipeline_data.get("targets", [])
        if not targets:
            return make_empty_figure("No targets loaded")

        ra = np.array([t.get("ra", 0) for t in targets])
        dec = np.array([t.get("dec", 0) for t in targets])
        tic_ids = [t.get("tic_id", "") for t in targets]
        statuses = [t.get("status", "processed") for t in targets]

        magnitudes = None
        if all("tmag" in t for t in targets):
            magnitudes = np.array([t["tmag"] for t in targets])

        return make_sky_map(ra, dec, tic_ids, statuses, magnitudes=magnitudes)

    # ------------------------------------------------------------------
    # Target selection — from table click or sky map click.
    # ------------------------------------------------------------------
    @app.callback(
        Output("selected-target", "data"),
        Input("candidate-table", "selected_rows"),
        Input("sky-map", "clickData"),
        State("candidate-table", "data"),
    )
    def select_target(
        selected_rows: list[int] | None,
        click_data: dict | None,
        table_data: list[dict],
    ) -> str | None:
        """Determine which target is selected (from table or sky map click)."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "candidate-table" and selected_rows:
            row_idx = selected_rows[0]
            if row_idx < len(table_data):
                return table_data[row_idx].get("tic_id")

        if trigger_id == "sky-map" and click_data:
            point = click_data.get("points", [{}])[0]
            custom = point.get("customdata")
            if custom:
                return custom

        return no_update

    # ------------------------------------------------------------------
    # Candidate selector dropdown — populates when a target is selected.
    # Shows all candidates for that target (multi-planet support).
    # ------------------------------------------------------------------
    @app.callback(
        Output("candidate-selector", "options"),
        Output("candidate-selector", "value"),
        Input("selected-target", "data"),
        State("pipeline-data", "data"),
    )
    def update_candidate_selector(
        selected_tic: str | None,
        pipeline_data: dict[str, Any],
    ) -> tuple[list[dict], str | None]:
        """Populate the candidate dropdown for the selected target."""
        if not selected_tic:
            return [], None

        target_candidates = _candidates_for_target(selected_tic, pipeline_data)
        if not target_candidates:
            return [], None

        options = []
        for i, c in enumerate(target_candidates):
            label = c.get("name", "")
            if not label:
                label = f"P={c['period']:.3f} d"
            label += f"  (depth={c['depth'] * 100:.3f}%, SNR={c['snr']:.1f})"
            options.append({"label": label, "value": i})

        # Default to the first candidate
        return options, 0

    # ------------------------------------------------------------------
    # Light curve plot — updates on target, candidate, or model toggle.
    # ------------------------------------------------------------------
    @app.callback(
        Output("lightcurve-plot", "figure"),
        Input("selected-target", "data"),
        Input("candidate-selector", "value"),
        Input("show-model-toggle", "value"),
        State("pipeline-data", "data"),
    )
    def update_lightcurve(
        selected_tic: str | None,
        candidate_idx: int | None,
        show_model: bool,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the light curve plot for the selected target and candidate."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its light curve")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(f"No light curve data for {selected_tic}")

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        # Resolve the selected candidate (if any).
        # Guard against stale index from a previous target with more candidates.
        candidate = None
        if show_model:
            target_candidates = _candidates_for_target(selected_tic, pipeline_data)
            if target_candidates:
                if candidate_idx is None or candidate_idx >= len(target_candidates):
                    candidate_idx = 0
                candidate = _candidate_from_dict(target_candidates[candidate_idx])

        return make_lightcurve_plot(
            time=time,
            flux_raw=None,
            flux_processed=flux,
            candidate=candidate,
            show_model=show_model,
        )

    # ------------------------------------------------------------------
    # Phase-folded plot — updates on target or candidate selection.
    # ------------------------------------------------------------------
    @app.callback(
        Output("phase-plot", "figure"),
        Input("selected-target", "data"),
        Input("candidate-selector", "value"),
        State("pipeline-data", "data"),
    )
    def update_phase_plot(
        selected_tic: str | None,
        candidate_idx: int | None,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the phase-folded plot for the selected target and candidate."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its phase diagram")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(f"No data for {selected_tic}")

        target_candidates = _candidates_for_target(selected_tic, pipeline_data)
        if not target_candidates:
            return make_empty_figure(f"No transit candidate for {selected_tic}")

        if candidate_idx is None:
            candidate_idx = 0
        if candidate_idx >= len(target_candidates):
            candidate_idx = 0

        candidate = _candidate_from_dict(target_candidates[candidate_idx])
        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        return make_phase_plot(time, flux, candidate)

    # ------------------------------------------------------------------
    # CSV export — downloads current table contents.
    # ------------------------------------------------------------------
    @app.callback(
        Output("export-download", "data"),
        Input("export-btn", "n_clicks"),
        State("candidate-table", "data"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks: int, table_data: list[dict]) -> Any:
        """Export the current candidate table as a CSV download."""
        if not table_data:
            return no_update

        df = pd.DataFrame(table_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        return dict(
            content=csv_buffer.getvalue(),
            filename="exohunter_candidates.csv",
        )
