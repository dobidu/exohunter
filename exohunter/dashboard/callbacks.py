"""Dash callbacks for interactive behaviour.

Data flow:
    1. On page load, the data-source dropdown is populated with
       available sectors from data/results/ plus the built-in demo.
    2. Selecting a data source loads its data into pipeline-data Store.
    3. Filters (classification, period, SNR, status) update the table.
    4. Clicking a table row or sky map point selects a target.
    5. The selected target populates the candidate selector and plots.
    6. The new-candidates panel auto-updates with NEW_CANDIDATE entries.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, html, no_update

from exohunter import config
from exohunter.dashboard.figures import (
    make_empty_figure,
    make_lightcurve_plot,
    make_odd_even_plot,
    make_periodogram_plot,
    make_phase_plot,
    make_sky_map,
)
from exohunter.detection.bls import TransitCandidate
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_from_dict(data: dict) -> TransitCandidate:
    """Reconstruct a ``TransitCandidate`` from a JSON-serializable dict."""
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


def _scan_available_sectors() -> list[dict]:
    """Scan data/results/ for processed sector CSV files.

    Returns:
        List of dicts with ``label`` and ``value`` for the dropdown.
    """
    results_dir = config.RESULTS_DIR
    options = []

    if results_dir.exists():
        for csv_path in sorted(results_dir.glob("sector_*_candidates.csv")):
            # Extract sector number from filename: sector_56_candidates.csv → 56
            name = csv_path.stem  # "sector_56_candidates"
            parts = name.split("_")
            if len(parts) >= 2:
                sector_num = parts[1]
                options.append({
                    "label": f"Sector {sector_num} (batch results)",
                    "value": f"sector:{csv_path.name}",
                })

        # Also check for summary files without _candidates suffix
        for csv_path in sorted(results_dir.glob("sector_*.csv")):
            if "_candidates" in csv_path.stem:
                continue
            name = csv_path.stem
            parts = name.split("_")
            if len(parts) >= 2:
                sector_num = parts[1]
                key = f"sector_summary:{csv_path.name}"
                # Only add if we don't already have a _candidates version
                existing = [o["value"] for o in options]
                if not any(f"sector:{name.replace(name, '')}" in v for v in existing):
                    if not any(f"sector:sector_{sector_num}_candidates.csv" in v for v in existing):
                        options.append({
                            "label": f"Sector {sector_num} (summary only)",
                            "value": key,
                        })

    return options


def _load_sector_data(filename: str) -> dict:
    """Load batch results from a sector CSV into pipeline-data format.

    Args:
        filename: CSV filename within data/results/.

    Returns:
        Dict in the pipeline-data Store schema.
    """
    csv_path = config.RESULTS_DIR / filename

    if not csv_path.exists():
        logger.warning("Sector file not found: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path)

    candidates = []
    targets = []
    seen_tics: set[str] = set()

    for _, row in df.iterrows():
        tic_id = str(row.get("tic_id", ""))
        if not tic_id:
            continue

        period = row.get("period")
        depth = row.get("depth")
        snr = row.get("snr")

        # Skip entries without detection data
        if pd.isna(period) or pd.isna(snr):
            continue

        xmatch_class = str(row.get("xmatch_class", ""))
        status = str(row.get("status", "candidate"))
        score = float(row.get("score", 0)) if "score" in row and not pd.isna(row.get("score")) else 0.0

        candidates.append({
            "tic_id": tic_id,
            "name": "",
            "period": float(period),
            "epoch": float(row.get("epoch", 0)) if not pd.isna(row.get("epoch", float("nan"))) else 0.0,
            "duration": float(row.get("duration", 0)) if not pd.isna(row.get("duration", float("nan"))) else 0.0,
            "depth": float(depth),
            "snr": float(snr),
            "bls_power": 0,
            "n_transits": int(row.get("n_transits", 0)) if not pd.isna(row.get("n_transits", float("nan"))) else 0,
            "status": status,
            "xmatch_class": xmatch_class,
            "score": score,
            "flags": str(row.get("flags", "")) if not pd.isna(row.get("flags", "")) else "",
        })

        if tic_id not in seen_tics:
            seen_tics.add(tic_id)
            targets.append({
                "tic_id": tic_id,
                "ra": 0.0,
                "dec": 0.0,
                "status": xmatch_class.lower() if xmatch_class else status,
            })

    return {
        "targets": targets,
        "candidates": candidates,
        "lightcurves": {},
    }


_XMATCH_BADGE_COLORS = {
    "NEW_CANDIDATE": "success",
    "KNOWN_MATCH": "info",
    "KNOWN_TOI": "warning",
    "HARMONIC": "danger",
}

_XMATCH_LABELS = {
    "NEW_CANDIDATE": "NEW",
    "KNOWN_MATCH": "MATCH",
    "KNOWN_TOI": "TOI",
    "HARMONIC": "HARM",
}


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_callbacks(app: Dash) -> None:
    """Register all Dash callbacks on the given app instance."""

    # ------------------------------------------------------------------
    # Data source dropdown — populated on load with available sectors.
    # ------------------------------------------------------------------
    @app.callback(
        Output("data-source-selector", "options"),
        Output("data-source-selector", "value"),
        Input("pipeline-data", "id"),
    )
    def populate_data_sources(_: str) -> tuple[list[dict], str]:
        """Populate the data source dropdown with demo + available sectors."""
        options = [{"label": "Demo — TOI-700 (synthetic)", "value": "demo"}]
        sector_options = _scan_available_sectors()
        options.extend(sector_options)

        # Default to demo
        return options, "demo"

    # ------------------------------------------------------------------
    # Switch data source — loads sector CSV or demo data into Store.
    # ------------------------------------------------------------------
    @app.callback(
        Output("pipeline-data", "data"),
        Input("data-source-selector", "value"),
        State("pipeline-data", "data"),
    )
    def switch_data_source(
        source: str | None,
        current_data: dict[str, Any],
    ) -> dict:
        """Load data from the selected source into the pipeline-data Store."""
        if not source or source == "demo":
            # Return current data (demo was already injected at startup)
            if current_data and current_data.get("candidates"):
                return current_data
            # If empty, generate demo data
            from scripts.run_dashboard import generate_demo_data
            return generate_demo_data()

        if source.startswith("sector:"):
            filename = source.split(":", 1)[1]
            return _load_sector_data(filename)

        if source.startswith("sector_summary:"):
            filename = source.split(":", 1)[1]
            return _load_sector_data(filename)

        return current_data

    # ------------------------------------------------------------------
    # New candidates highlight panel.
    # ------------------------------------------------------------------
    @app.callback(
        Output("new-candidates-panel", "children"),
        Input("pipeline-data", "data"),
    )
    def update_new_candidates_panel(pipeline_data: dict[str, Any]) -> list:
        """Update the highlight panel showing uncatalogued candidates."""
        candidates = pipeline_data.get("candidates", [])
        new_ones = [
            c for c in candidates
            if c.get("xmatch_class") == "NEW_CANDIDATE"
        ]

        if not new_ones:
            return [html.P(
                "No new uncatalogued candidates in current data.",
                className="text-muted mb-0",
            )]

        # Sort by score (or SNR as fallback)
        new_ones.sort(key=lambda c: c.get("score", c.get("snr", 0)), reverse=True)

        items = []
        for c in new_ones[:10]:  # Show top 10
            badge = dbc.Badge("NEW", color="success", className="me-2")
            score_val = c.get("score", 0)
            items.append(html.Div([
                badge,
                html.Strong(c.get("tic_id", ""), className="me-2"),
                html.Span(
                    f"P={c.get('period', 0):.4f} d, "
                    f"depth={c.get('depth', 0) * 100:.3f}%, "
                    f"SNR={c.get('snr', 0):.1f}, "
                    f"score={score_val:.1f}",
                    className="text-muted",
                ),
            ], className="mb-1"))

        header = html.P(
            f"{len(new_ones)} uncatalogued candidate(s) found!",
            className="fw-bold mb-2",
            style={"color": "#00ff88"},
        )

        return [header] + items

    # ------------------------------------------------------------------
    # Candidate table — filtered by all sidebar controls.
    # ------------------------------------------------------------------
    @app.callback(
        Output("candidate-table", "data"),
        Input("period-range", "value"),
        Input("min-snr", "value"),
        Input("status-filter", "value"),
        Input("xmatch-filter", "value"),
        Input("pipeline-data", "data"),
    )
    def update_candidate_table(
        period_range: list[float],
        min_snr: float,
        status_filter: list[str],
        xmatch_filter: list[str],
        pipeline_data: dict[str, Any],
    ) -> list[dict]:
        """Filter and update the candidate table."""
        candidates = pipeline_data.get("candidates", [])
        if not candidates:
            return []

        filtered = []
        for c in candidates:
            period = c.get("period", 0)
            snr = c.get("snr", 0)
            status = c.get("status", "candidate").lower()
            xmatch = c.get("xmatch_class", "")

            if period < period_range[0] or period > period_range[1]:
                continue
            if snr < min_snr:
                continue
            if status not in status_filter:
                continue
            # Apply classification filter (skip if xmatch is empty —
            # entries without classification always pass)
            if xmatch and xmatch_filter and xmatch not in xmatch_filter:
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
                "xmatch_class": xmatch,
                "status": c.get("status", "Candidate").capitalize(),
                "flags": c.get("flags", ""),
            })

        return filtered

    # ------------------------------------------------------------------
    # Sky map.
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
        """Determine which target is selected."""
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
    # Candidate selector dropdown — multi-planet support.
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
            xmatch = c.get("xmatch_class", "")
            badge = f" [{xmatch}]" if xmatch else ""
            label += f"  (SNR={c['snr']:.1f}{badge})"
            options.append({"label": label, "value": i})

        return options, 0

    # ------------------------------------------------------------------
    # Light curve plot.
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
        """Update the light curve plot for the selected target."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its light curve")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(
                f"Light curve not available for {selected_tic}\n"
                f"(batch mode — only candidate table and sky map are shown)"
            )

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

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
    # Phase-folded plot.
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
        """Update the phase-folded plot for the selected target."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its phase diagram")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(
                f"Phase diagram not available for {selected_tic}\n"
                f"(batch mode — light curves are not stored)"
            )

        target_candidates = _candidates_for_target(selected_tic, pipeline_data)
        if not target_candidates:
            return make_empty_figure(f"No transit candidate for {selected_tic}")

        if candidate_idx is None or candidate_idx >= len(target_candidates):
            candidate_idx = 0

        candidate = _candidate_from_dict(target_candidates[candidate_idx])
        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        return make_phase_plot(time, flux, candidate)

    # ------------------------------------------------------------------
    # BLS periodogram plot.
    # ------------------------------------------------------------------
    @app.callback(
        Output("periodogram-plot", "figure"),
        Input("selected-target", "data"),
        Input("candidate-selector", "value"),
        State("pipeline-data", "data"),
    )
    def update_periodogram(
        selected_tic: str | None,
        candidate_idx: int | None,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the BLS periodogram for the selected target."""
        if not selected_tic:
            return make_empty_figure("Select a target to view its periodogram")

        # Check if BLS periodogram data is available
        bls_data = pipeline_data.get("bls_periodograms", {}).get(selected_tic)
        if not bls_data:
            return make_empty_figure(
                f"BLS periodogram not available for {selected_tic}\n"
                f"(run inspect_candidate.py for full diagnostics)"
            )

        target_candidates = _candidates_for_target(selected_tic, pipeline_data)
        if not target_candidates:
            return make_empty_figure(f"No candidate for {selected_tic}")

        if candidate_idx is None or candidate_idx >= len(target_candidates):
            candidate_idx = 0

        candidate = _candidate_from_dict(target_candidates[candidate_idx])
        periods = np.array(bls_data["periods"])
        power = np.array(bls_data["power"])

        return make_periodogram_plot(periods, power, candidate)

    # ------------------------------------------------------------------
    # Odd-even transit comparison plot.
    # ------------------------------------------------------------------
    @app.callback(
        Output("odd-even-plot", "figure"),
        Input("selected-target", "data"),
        Input("candidate-selector", "value"),
        State("pipeline-data", "data"),
    )
    def update_odd_even(
        selected_tic: str | None,
        candidate_idx: int | None,
        pipeline_data: dict[str, Any],
    ) -> Any:
        """Update the odd-even transit comparison plot."""
        if not selected_tic:
            return make_empty_figure("Select a target to view odd-even comparison")

        lc_data = pipeline_data.get("lightcurves", {}).get(selected_tic)
        if not lc_data:
            return make_empty_figure(
                f"Odd-even comparison not available for {selected_tic}\n"
                f"(requires light curve data)"
            )

        target_candidates = _candidates_for_target(selected_tic, pipeline_data)
        if not target_candidates:
            return make_empty_figure(f"No candidate for {selected_tic}")

        if candidate_idx is None or candidate_idx >= len(target_candidates):
            candidate_idx = 0

        candidate = _candidate_from_dict(target_candidates[candidate_idx])
        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        return make_odd_even_plot(time, flux, candidate)

    # ------------------------------------------------------------------
    # CSV export.
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
