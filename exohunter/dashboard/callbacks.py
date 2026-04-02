"""Dash callbacks for interactive behaviour.

Each callback connects user interactions (clicks, slider changes)
to updates in the dashboard visualizations.

Data flow:
    1. On page load, ``pipeline-data`` store is populated with demo data.
    2. Filters update the candidate table.
    3. Clicking a row in the table (or a point on the sky map) selects
       a target and updates the light curve and phase plots.
    4. The export button triggers a CSV download.
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
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def register_callbacks(app: Dash) -> None:
    """Register all Dash callbacks on the given app instance.

    Args:
        app: The Dash application.
    """

    @app.callback(
        Output("candidate-table", "data"),
        Input("period-range", "value"),
        Input("min-snr", "value"),
        Input("status-filter", "value"),
        State("pipeline-data", "data"),
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
                "period": c.get("period", 0),
                "epoch": c.get("epoch", 0),
                "duration": c.get("duration", 0),
                "depth_pct": c.get("depth", 0) * 100,
                "snr": c.get("snr", 0),
                "status": c.get("status", "Candidate"),
                "flags": c.get("flags", ""),
            })

        return filtered

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

        return make_sky_map(ra, dec, tic_ids, statuses)

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

    @app.callback(
        Output("lightcurve-plot", "figure"),
        Input("selected-target", "data"),
        Input("show-model-toggle", "value"),
        State("pipeline-data", "data"),
    )
    def update_lightcurve(
        selected_tic: str | None,
        show_model: bool,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the light curve plot for the selected target."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its light curve")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(f"No light curve data for {selected_tic}")

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        # Find the candidate for this target
        candidate_data = None
        if show_model:
            for c in pipeline_data.get("candidates", []):
                if c.get("tic_id") == selected_tic:
                    candidate_data = c
                    break

        # Build a minimal TransitCandidate for the figure function
        candidate = None
        if candidate_data:
            from exohunter.detection.bls import TransitCandidate
            candidate = TransitCandidate(
                tic_id=candidate_data["tic_id"],
                period=candidate_data["period"],
                epoch=candidate_data["epoch"],
                duration=candidate_data["duration"],
                depth=candidate_data["depth"],
                snr=candidate_data["snr"],
                bls_power=candidate_data.get("bls_power", 0),
            )

        return make_lightcurve_plot(
            time=time,
            flux_raw=None,
            flux_processed=flux,
            candidate=candidate,
            show_model=show_model,
        )

    @app.callback(
        Output("phase-plot", "figure"),
        Input("selected-target", "data"),
        State("pipeline-data", "data"),
    )
    def update_phase_plot(
        selected_tic: str | None,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the phase-folded plot for the selected target."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its phase diagram")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(f"No data for {selected_tic}")

        # Find the candidate
        candidate_data = None
        for c in pipeline_data.get("candidates", []):
            if c.get("tic_id") == selected_tic:
                candidate_data = c
                break

        if not candidate_data:
            return make_empty_figure(f"No transit candidate for {selected_tic}")

        from exohunter.detection.bls import TransitCandidate
        candidate = TransitCandidate(
            tic_id=candidate_data["tic_id"],
            period=candidate_data["period"],
            epoch=candidate_data["epoch"],
            duration=candidate_data["duration"],
            depth=candidate_data["depth"],
            snr=candidate_data["snr"],
            bls_power=candidate_data.get("bls_power", 0),
        )

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        return make_phase_plot(time, flux, candidate)

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
