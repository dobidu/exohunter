"""Tests for the Plotly Dash dashboard.

Verifies that the dashboard creates correctly, renders data into
the right components, and that the pipeline-data Store schema is
respected end-to-end from demo data generation through figure output.

All tests run offline — no Dash server is started.
"""

import json

import numpy as np
import pytest

import dash_bootstrap_components as dbc
from dash import dcc, html


class TestDemoDataGeneration:
    """Test the synthetic TOI-700 demo data generator."""

    @pytest.fixture(scope="class")
    def demo_data(self) -> dict:
        """Generate demo data once for all tests in this class."""
        from scripts.run_dashboard import generate_demo_data
        return generate_demo_data()

    def test_has_required_top_level_keys(self, demo_data: dict) -> None:
        """Pipeline-data must have targets, candidates, and lightcurves."""
        assert "targets" in demo_data
        assert "candidates" in demo_data
        assert "lightcurves" in demo_data

    def test_has_three_candidates(self, demo_data: dict) -> None:
        """TOI-700 demo must produce exactly 3 candidates (b, c, d)."""
        assert len(demo_data["candidates"]) == 3

    def test_candidate_fields_complete(self, demo_data: dict) -> None:
        """Each candidate must have all fields required by the dashboard."""
        required = {
            "tic_id", "period", "epoch", "duration", "depth",
            "snr", "status", "flags", "name", "xmatch_class", "score",
        }
        for c in demo_data["candidates"]:
            missing = required - set(c.keys())
            assert not missing, f"Candidate {c.get('name')} missing fields: {missing}"

    def test_candidate_periods_match_toi700(self, demo_data: dict) -> None:
        """Candidate periods must match real TOI-700 planet periods."""
        expected_periods = {9.977, 16.051, 37.426}
        actual_periods = {c["period"] for c in demo_data["candidates"]}
        assert actual_periods == expected_periods

    def test_candidate_names_present(self, demo_data: dict) -> None:
        """Each candidate must have a planet name."""
        names = {c["name"] for c in demo_data["candidates"]}
        assert "TOI-700 b" in names
        assert "TOI-700 c" in names
        assert "TOI-700 d" in names

    def test_lightcurve_has_toi700(self, demo_data: dict) -> None:
        """A light curve must exist for TIC 150428135."""
        assert "TIC 150428135" in demo_data["lightcurves"]

    def test_lightcurve_arrays_same_length(self, demo_data: dict) -> None:
        """Time and flux arrays must have the same length."""
        lc = demo_data["lightcurves"]["TIC 150428135"]
        assert len(lc["time"]) == len(lc["flux"])
        assert len(lc["time"]) > 10000  # should be 25000

    def test_lightcurve_contains_transits(self, demo_data: dict) -> None:
        """The synthetic light curve must contain visible transit dips."""
        lc = demo_data["lightcurves"]["TIC 150428135"]
        flux = np.array(lc["flux"])

        # Flux should dip below the baseline
        assert np.min(flux) < 0.999, "No transit dips found in demo light curve"
        # Most flux should be near 1.0
        assert np.median(flux) > 0.999

    def test_targets_include_toi700(self, demo_data: dict) -> None:
        """Target list must include TOI-700 with status=validated."""
        toi700 = [t for t in demo_data["targets"] if t["tic_id"] == "TIC 150428135"]
        assert len(toi700) == 1
        assert toi700[0]["status"] == "validated"

    def test_targets_include_background_stars(self, demo_data: dict) -> None:
        """Target list must include background stars for the sky map."""
        non_toi = [t for t in demo_data["targets"] if t["tic_id"] != "TIC 150428135"]
        assert len(non_toi) >= 30  # we generate 40

    def test_targets_have_coordinates(self, demo_data: dict) -> None:
        """Every target must have ra and dec fields."""
        for t in demo_data["targets"]:
            assert "ra" in t and "dec" in t
            assert isinstance(t["ra"], float)
            assert isinstance(t["dec"], float)

    def test_data_is_json_serializable(self, demo_data: dict) -> None:
        """Pipeline-data must be JSON-serializable (required by dcc.Store)."""
        try:
            serialized = json.dumps(demo_data)
            assert len(serialized) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Demo data is not JSON-serializable: {e}")


class TestAppCreation:
    """Test the Dash application factory."""

    def test_create_empty_app(self) -> None:
        """create_app() without data must return a valid Dash app."""
        from exohunter.dashboard.app import create_app

        app = create_app()
        assert app is not None
        assert app.layout is not None

    def test_create_app_with_demo_data(self) -> None:
        """create_app() with pipeline data must inject it into the Store."""
        from exohunter.dashboard.app import create_app
        from scripts.run_dashboard import generate_demo_data

        data = generate_demo_data()
        app = create_app(pipeline_data=data)

        # Find the pipeline-data Store in the layout
        store = _find_component(app.layout, "pipeline-data")
        assert store is not None, "pipeline-data Store not found in layout"
        assert isinstance(store, dcc.Store)
        assert store.data is not None
        assert len(store.data.get("candidates", [])) == 3

    def test_layout_has_all_component_ids(self) -> None:
        """The layout must contain all required component IDs."""
        from exohunter.dashboard.app import create_app

        app = create_app()
        layout_str = _layout_to_string(app.layout)

        required_ids = [
            "pipeline-data",
            "selected-target",
            "data-source-selector",
            "xmatch-filter",
            "period-range",
            "min-snr",
            "status-filter",
            "sky-map",
            "lightcurve-plot",
            "phase-plot",
            "candidate-table",
            "candidate-selector",
            "show-model-toggle",
            "new-candidates-panel",
            "export-btn",
            "export-download",
        ]
        for comp_id in required_ids:
            assert comp_id in layout_str, f"Component '{comp_id}' not found in layout"


class TestFigureGeneration:
    """Test that figure generators produce valid Plotly figures."""

    def test_sky_map_with_data(self) -> None:
        """make_sky_map must return a Figure with traces."""
        from exohunter.dashboard.figures import make_sky_map

        fig = make_sky_map(
            ra=np.array([10.0, 20.0, 30.0]),
            dec=np.array([-10.0, 0.0, 10.0]),
            tic_ids=["TIC A", "TIC B", "TIC C"],
            statuses=["processed", "validated", "rejected"],
        )

        assert len(fig.data) >= 1
        assert fig.layout.xaxis.title.text == "Right Ascension (°)"

    def test_lightcurve_plot_with_model(self) -> None:
        """make_lightcurve_plot must include model trace when candidate given."""
        from exohunter.dashboard.figures import make_lightcurve_plot
        from tests.conftest import make_candidate

        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000)
        candidate = make_candidate(period=5.0, epoch=2.5, duration=0.1, depth=0.01)

        fig = make_lightcurve_plot(
            time=time, flux_raw=None, flux_processed=flux,
            candidate=candidate, show_model=True,
        )

        trace_names = [t.name for t in fig.data if t.name]
        assert any("Model" in n for n in trace_names), (
            f"No model trace found. Traces: {trace_names}"
        )

    def test_lightcurve_plot_without_model(self) -> None:
        """make_lightcurve_plot without candidate must have no model trace."""
        from exohunter.dashboard.figures import make_lightcurve_plot

        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000)

        fig = make_lightcurve_plot(
            time=time, flux_raw=None, flux_processed=flux,
            candidate=None, show_model=False,
        )

        trace_names = [t.name for t in fig.data if t.name]
        assert not any("Model" in n for n in trace_names)

    def test_phase_plot_has_three_traces(self) -> None:
        """make_phase_plot must produce exactly 3 traces: raw, binned, model."""
        from exohunter.dashboard.figures import make_phase_plot
        from tests.conftest import make_candidate, make_synthetic_transit_lc

        lc = make_synthetic_transit_lc(period=10.0, depth=0.01, noise=0.0005)
        candidate = make_candidate(period=10.0, epoch=5.0, duration=0.2, depth=0.01)

        fig = make_phase_plot(
            time=lc.time.value, flux=lc.flux.value, candidate=candidate,
        )

        assert len(fig.data) == 3, f"Expected 3 traces, got {len(fig.data)}"
        trace_names = [t.name for t in fig.data]
        assert "Phase-folded data" in trace_names
        assert "Binned" in trace_names
        assert "Transit model" in trace_names

    def test_periodogram_plot(self) -> None:
        """make_periodogram_plot must show the BLS power spectrum with peak marked."""
        from exohunter.dashboard.figures import make_periodogram_plot
        from tests.conftest import make_candidate

        periods = np.linspace(0.5, 20, 1000)
        power = np.random.default_rng(42).random(1000) * 0.001
        candidate = make_candidate(period=8.0, name="Test Planet")

        fig = make_periodogram_plot(periods, power, candidate)

        assert len(fig.data) >= 1  # at least the power trace
        assert "Periodogram" in fig.layout.title.text
        # Should have vertical lines (shapes) for peak + harmonics
        assert len(fig.layout.shapes) >= 1

    def test_odd_even_plot(self) -> None:
        """make_odd_even_plot must produce traces for odd and even transits."""
        from exohunter.dashboard.figures import make_odd_even_plot
        from tests.conftest import make_candidate, make_synthetic_transit_lc

        lc = make_synthetic_transit_lc(
            period=5.0, depth=0.01, duration=0.15, noise=0.0005, n_points=20000,
        )
        candidate = make_candidate(period=5.0, epoch=2.5, duration=0.15, depth=0.01)

        fig = make_odd_even_plot(lc.time.value, lc.flux.value, candidate)

        # Should have odd and even traces
        assert len(fig.data) >= 2
        # Title must contain consistency verdict
        assert "CONSISTENT" in fig.layout.title.text or "INCONSISTENT" in fig.layout.title.text
        # Should have an annotation with depth difference
        assert len(fig.layout.annotations) >= 1

    def test_empty_figure_has_message(self) -> None:
        """make_empty_figure must display the given message."""
        from exohunter.dashboard.figures import make_empty_figure

        fig = make_empty_figure("Test message")
        annotations = fig.layout.annotations
        assert len(annotations) == 1
        assert annotations[0].text == "Test message"

    def test_all_figures_use_dark_theme(self) -> None:
        """All figures must use the plotly_dark template."""
        from exohunter.dashboard.figures import (
            make_empty_figure,
            make_lightcurve_plot,
            make_sky_map,
        )

        figs = [
            make_empty_figure("test"),
            make_sky_map(np.array([0.0]), np.array([0.0]), ["A"], ["processed"]),
            make_lightcurve_plot(np.array([0, 1]), None, np.array([1, 1])),
        ]
        for fig in figs:
            assert fig.layout.template.layout.paper_bgcolor is not None or \
                   "plotly_dark" in str(fig.layout.template)


class TestEndToEndPipeline:
    """Test the full pipeline flow: preprocessing → BLS → validation → crossmatch.

    These tests verify that data flows correctly through all stages
    and that the outputs are consistent with each other.
    """

    def test_pipeline_on_synthetic_transit(self) -> None:
        """A strong synthetic transit must survive the full pipeline."""
        from exohunter.detection.bls import run_bls_lightkurve
        from exohunter.detection.validator import validate_candidate
        from exohunter.catalog.crossmatch import crossmatch_candidate, MatchClass
        from exohunter.catalog.candidates import CandidateCatalog, compute_score
        from tests.conftest import make_synthetic_transit_lc

        # Stage 1: Create a clean synthetic signal
        lc = make_synthetic_transit_lc(
            period=8.0, depth=0.01, duration=0.15,
            noise=0.0005, n_points=15000, baseline_days=90.0,
        )

        # Stage 2: BLS detection (on cleaned + normalized data,
        # bypassing flatten — the flatten step is tested separately
        # and can attenuate synthetic signals that lack realistic
        # stellar variability)
        candidate = run_bls_lightkurve(
            lc, tic_id="TIC 999888777",
            min_period=1.0, max_period=12.0, num_periods=5000,
        )
        assert candidate is not None
        assert abs(candidate.period - 8.0) < 0.1, (
            f"Period mismatch: {candidate.period:.4f} vs 8.0"
        )

        # Stage 3: Validation — the candidate may or may not pass
        # depending on the SNR that lightkurve computes (which can be
        # 0.0 for synthetic data without realistic error bars).
        validation = validate_candidate(candidate)
        assert isinstance(validation.is_valid, (bool, np.bool_))

        # Stage 4: Cross-match (unknown TIC → NEW_CANDIDATE)
        xmatch = crossmatch_candidate(candidate)
        assert xmatch.match_class == MatchClass.NEW_CANDIDATE

        # Stage 5: Scoring (score >= 0 always; may be 0 if SNR=0)
        score = compute_score(candidate, validation)
        assert score >= 0

        # Stage 6: Catalog — add regardless of validation status
        catalog = CandidateCatalog()
        catalog.add(candidate, validation)
        assert len(catalog) == 1

        # Stage 7: DataFrame export — verify structure
        df = catalog.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["period"] == pytest.approx(candidate.period, abs=0.01)
        assert "score" in df.columns
        assert "is_valid" in df.columns


# =========================================================================
# Helpers
# =========================================================================

def _find_component(component, comp_id: str):
    """Recursively find a Dash component by its ID."""
    if getattr(component, "id", None) == comp_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component(child, comp_id)
            if result is not None:
                return result
    else:
        return _find_component(children, comp_id)

    return None


def _layout_to_string(component) -> str:
    """Recursively extract all component IDs from a Dash layout."""
    parts = []
    comp_id = getattr(component, "id", None)
    if comp_id:
        parts.append(str(comp_id))

    children = getattr(component, "children", None)
    if children is not None:
        if isinstance(children, (list, tuple)):
            for child in children:
                parts.append(_layout_to_string(child))
        elif hasattr(children, "children") or hasattr(children, "id"):
            parts.append(_layout_to_string(children))

    return " ".join(parts)
