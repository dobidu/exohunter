"""Dashboard page layouts for ExoHunter.

Defines the visual structure of the Dash application using
dash-bootstrap-components (dbc) with the DARKLY theme.

Layout structure:
    ┌─────────────────────────────────────────────┐
    │  Navbar                                     │
    ├──────────┬──────────────────────────────────┤
    │ Sidebar  │  Sky Map (Panel 1)               │
    │ (filters)├──────────────────────────────────┤
    │          │  Light Curve (Panel 2)           │
    │          ├──────────────────────────────────┤
    │          │  Phase Diagram (Panel 3)         │
    ├──────────┴──────────────────────────────────┤
    │  Candidate Table                            │
    └─────────────────────────────────────────────┘
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


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
                dbc.NavItem(dbc.NavLink("About", href="#about")),
            ], navbar=True),
        ]),
        color="primary",
        dark=True,
        className="mb-3",
    )


def make_sidebar() -> dbc.Card:
    """Create the sidebar with filter controls."""
    return dbc.Card([
        dbc.CardHeader("Filters"),
        dbc.CardBody([
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
            html.Label("Status", className="fw-bold mb-1"),
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

            html.Hr(),

            # Pipeline progress indicator
            html.Label("Pipeline Status", className="fw-bold mb-1"),
            html.Div(id="pipeline-status", children=[
                dbc.Badge("Ready", color="info", className="me-1"),
            ]),
        ]),
    ], className="h-100")


def make_main_content() -> html.Div:
    """Create the main content area with three visualization panels."""
    return html.Div([
        # Panel 1 — Sky Map
        dbc.Card([
            dbc.CardHeader("Sky Map — Target Overview"),
            dbc.CardBody(dcc.Graph(id="sky-map", style={"height": "500px"})),
        ], className="mb-3"),

        # Panel 2 — Light Curve
        dbc.Card([
            dbc.CardHeader([
                "Light Curve",
                dbc.Switch(
                    id="show-model-toggle",
                    label="Show Transit Model",
                    value=True,
                    className="float-end",
                ),
            ]),
            dbc.CardBody(dcc.Graph(id="lightcurve-plot", style={"height": "450px"})),
        ], className="mb-3"),

        # Panel 3 — Phase Diagram
        dbc.Card([
            dbc.CardHeader("Phase-Folded Light Curve"),
            dbc.CardBody(dcc.Graph(id="phase-plot", style={"height": "400px"})),
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
                        "if": {"filter_query": '{status} = "Validated"'},
                        "backgroundColor": "rgba(0, 128, 0, 0.15)",
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


def make_layout() -> dbc.Container:
    """Assemble the complete dashboard layout.

    Returns:
        A ``dbc.Container`` with all components arranged in the
        dashboard grid.
    """
    return dbc.Container([
        make_navbar(),

        # Hidden store for pipeline data (shared between callbacks)
        dcc.Store(id="pipeline-data", data={}),
        dcc.Store(id="selected-target", data=None),

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
