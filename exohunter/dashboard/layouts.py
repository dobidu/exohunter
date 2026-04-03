"""Dashboard page layouts for ExoHunter.

Defines the visual structure of the Dash application using
dash-bootstrap-components (dbc) with the DARKLY theme.

Layout structure:
    ┌─────────────────────────────────────────────────────┐
    │  Navbar                                             │
    ├──────────┬──────────────────────────────────────────┤
    │ Sidebar  │  New Candidates Highlight Panel          │
    │ (filters)├──────────────────────────────────────────┤
    │          │  Sky Map (Panel 1)                       │
    │          ├──────────────────────────────────────────┤
    │          │  Light Curve (Panel 2)                   │
    │          ├──────────────────────────────────────────┤
    │          │  Phase Diagram (Panel 3)                 │
    ├──────────┴──────────────────────────────────────────┤
    │  Candidate Table                                    │
    └─────────────────────────────────────────────────────┘
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


# ---------------------------------------------------------------------------
# Badge colors for each cross-match classification
# ---------------------------------------------------------------------------
XMATCH_BADGE_COLORS: dict[str, str] = {
    "NEW_CANDIDATE": "success",
    "KNOWN_MATCH": "info",
    "KNOWN_TOI": "warning",
    "HARMONIC": "danger",
}


def make_navbar() -> dbc.Navbar:
    """Create the top navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                "ExoHunter — Exoplanet Transit Detection",
                className="ms-2",
                style={"fontWeight": "bold", "fontSize": "1.3rem"},
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True)),
            ], navbar=True),
        ]),
        color="primary",
        dark=True,
        className="mb-3",
    )


def make_sidebar() -> dbc.Card:
    """Create the sidebar with filter controls."""
    return dbc.Card([
        dbc.CardHeader("Data & Filters"),
        dbc.CardBody([
            # Data source selector
            html.Label("Data Source", className="fw-bold mb-1"),
            dcc.Dropdown(
                id="data-source-selector",
                placeholder="Select data source...",
                clearable=False,
                style={"backgroundColor": "rgb(50,50,50)", "color": "white"},
                className="mb-2",
            ),

            html.Hr(),

            # Classification filter
            html.Label("Classification", className="fw-bold mb-1"),
            dbc.Checklist(
                id="xmatch-filter",
                options=[
                    {"label": html.Span([
                        dbc.Badge("NEW", color="success", className="me-1"),
                        " New Candidate",
                    ]), "value": "NEW_CANDIDATE"},
                    {"label": html.Span([
                        dbc.Badge("MATCH", color="info", className="me-1"),
                        " Known Match",
                    ]), "value": "KNOWN_MATCH"},
                    {"label": html.Span([
                        dbc.Badge("TOI", color="warning", className="me-1"),
                        " Known TOI",
                    ]), "value": "KNOWN_TOI"},
                    {"label": html.Span([
                        dbc.Badge("HARM", color="danger", className="me-1"),
                        " Harmonic",
                    ]), "value": "HARMONIC"},
                ],
                value=["NEW_CANDIDATE", "KNOWN_MATCH", "KNOWN_TOI", "HARMONIC"],
                className="mb-3",
            ),

            html.Hr(),

            # Period range filter
            html.Label("Period Range (days)", className="fw-bold mb-1"),
            dcc.RangeSlider(
                id="period-range",
                min=0.5,
                max=50.0,
                step=0.5,
                value=[0.5, 50.0],
                marks={i: str(i) for i in range(0, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),

            html.Hr(),

            # Minimum SNR filter
            html.Label("Minimum SNR", className="fw-bold mb-1"),
            dcc.Slider(
                id="min-snr",
                min=0,
                max=50,
                step=1,
                value=7,
                marks={0: "0", 7: "7", 20: "20", 50: "50"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),

            html.Hr(),

            # Status filter
            html.Label("Validation Status", className="fw-bold mb-1"),
            dbc.Checklist(
                id="status-filter",
                options=[
                    {"label": " Validated", "value": "validated"},
                    {"label": " Candidate", "value": "candidate"},
                    {"label": " Rejected", "value": "rejected"},
                ],
                value=["validated", "candidate"],
                className="mb-3",
            ),

            html.Hr(),

            # Export button
            dbc.Button(
                "Export CSV",
                id="export-btn",
                color="success",
                className="w-100 mt-2",
            ),
            dcc.Download(id="export-download"),
        ]),
    ], className="h-100")


def make_new_candidates_panel() -> dbc.Card:
    """Create the highlight panel for new uncatalogued candidates."""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Badge("NEW", color="success", className="me-2"),
            "New Candidates — Not in Any Catalog",
        ]),
        dbc.CardBody(
            html.Div(id="new-candidates-panel", children=[
                html.P(
                    "No new candidates yet. Process a sector to find them!",
                    className="text-muted mb-0",
                ),
            ]),
        ),
    ], className="mb-3", style={"borderColor": "#00ff88", "borderWidth": "1px"})


def make_main_content() -> html.Div:
    """Create the main content area with highlight panel and three viz panels."""
    return html.Div([
        # Highlight panel — new uncatalogued candidates
        make_new_candidates_panel(),

        # Panel 1 — Sky Map
        dbc.Card([
            dbc.CardHeader("Panel 1 — Sky Map"),
            dbc.CardBody(dcc.Graph(id="sky-map", style={"height": "500px"})),
        ], className="mb-3"),

        # Panel 2 — Light Curve
        dbc.Card([
            dbc.CardHeader([
                "Panel 2 — Light Curve",
                html.Div([
                    dbc.Switch(
                        id="show-model-toggle",
                        label="Show Transit Model",
                        value=True,
                        className="d-inline-block me-3",
                    ),
                    dcc.Dropdown(
                        id="candidate-selector",
                        placeholder="Select candidate...",
                        clearable=False,
                        style={
                            "width": "220px",
                            "display": "inline-block",
                            "verticalAlign": "middle",
                            "backgroundColor": "rgb(50, 50, 50)",
                            "color": "white",
                        },
                    ),
                ], className="float-end d-flex align-items-center gap-2"),
            ]),
            dbc.CardBody(dcc.Graph(id="lightcurve-plot", style={"height": "450px"})),
        ], className="mb-3"),

        # Panel 3 — Phase Diagram
        dbc.Card([
            dbc.CardHeader("Panel 3 — Phase-Folded Light Curve"),
            dbc.CardBody(dcc.Graph(id="phase-plot", style={"height": "400px"})),
        ], className="mb-3"),

        # Diagnostic panels — BLS periodogram and odd-even comparison
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Panel 4 — BLS Periodogram"),
                    dbc.CardBody(dcc.Graph(id="periodogram-plot", style={"height": "350px"})),
                ]),
                width=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Panel 5 — Odd vs Even Transits"),
                    dbc.CardBody(dcc.Graph(id="odd-even-plot", style={"height": "350px"})),
                ]),
                width=6,
            ),
        ], className="mb-3"),
    ])


def make_candidate_table() -> dbc.Card:
    """Create the candidate results table."""
    return dbc.Card([
        dbc.CardHeader("Candidate Table"),
        dbc.CardBody(
            dash_table.DataTable(
                id="candidate-table",
                columns=[
                    {"name": "TIC ID", "id": "tic_id"},
                    {"name": "Planet", "id": "name"},
                    {"name": "Period (d)", "id": "period", "type": "numeric",
                     "format": dash_table.Format.Format(precision=4)},
                    {"name": "Epoch (BTJD)", "id": "epoch", "type": "numeric",
                     "format": dash_table.Format.Format(precision=4)},
                    {"name": "Duration (d)", "id": "duration", "type": "numeric",
                     "format": dash_table.Format.Format(precision=4)},
                    {"name": "Depth (%)", "id": "depth_pct", "type": "numeric",
                     "format": dash_table.Format.Format(precision=4)},
                    {"name": "SNR", "id": "snr", "type": "numeric",
                     "format": dash_table.Format.Format(precision=1)},
                    {"name": "Score", "id": "score", "type": "numeric",
                     "format": dash_table.Format.Format(precision=1)},
                    {"name": "Classification", "id": "xmatch_class"},
                    {"name": "Status", "id": "status"},
                    {"name": "Flags", "id": "flags"},
                ],
                data=[],
                sort_action="native",
                filter_action="native",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "rgb(30, 30, 30)",
                    "fontWeight": "bold",
                    "color": "white",
                },
                style_cell={
                    "backgroundColor": "rgb(50, 50, 50)",
                    "color": "white",
                    "border": "1px solid rgb(70, 70, 70)",
                    "textAlign": "left",
                    "padding": "8px",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{xmatch_class} = "NEW_CANDIDATE"'},
                        "backgroundColor": "rgba(0, 255, 100, 0.20)",
                        "color": "#00ff88",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": '{xmatch_class} = "KNOWN_MATCH"'},
                        "backgroundColor": "rgba(0, 128, 255, 0.15)",
                    },
                    {
                        "if": {"filter_query": '{xmatch_class} = "KNOWN_TOI"'},
                        "backgroundColor": "rgba(255, 200, 0, 0.12)",
                    },
                    {
                        "if": {"filter_query": '{xmatch_class} = "HARMONIC"'},
                        "backgroundColor": "rgba(255, 100, 0, 0.12)",
                    },
                    {
                        "if": {"filter_query": '{status} = "Rejected"'},
                        "backgroundColor": "rgba(255, 0, 0, 0.1)",
                    },
                ],
                row_selectable="single",
            ),
        ),
    ], className="mb-3")


def make_layout(pipeline_data: dict | None = None) -> dbc.Container:
    """Assemble the complete dashboard layout.

    Args:
        pipeline_data: Optional initial data for the pipeline-data Store.
            If provided, all panels will render immediately on page load.

    Returns:
        A ``dbc.Container`` with all components arranged in the
        dashboard grid.
    """
    return dbc.Container([
        make_navbar(),

        # Hidden stores for shared state between callbacks
        dcc.Store(id="pipeline-data", data=pipeline_data or {}),
        dcc.Store(id="selected-target", data=None),
        dcc.Store(id="selected-candidate-idx", data=0),

        dbc.Row([
            # Sidebar
            dbc.Col(make_sidebar(), width=3),

            # Main content
            dbc.Col(make_main_content(), width=9),
        ], className="mb-3"),

        # Candidate table (full width)
        dbc.Row([
            dbc.Col(make_candidate_table(), width=12),
        ]),

    ], fluid=True)
