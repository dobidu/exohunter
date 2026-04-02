"""Main Dash application factory for ExoHunter.

Creates and configures the Plotly Dash app with the DARKLY Bootstrap
theme, registers all callbacks, and exposes a ``create_app`` function
that scripts can call to get a running server.

Usage::

    from exohunter.dashboard.app import create_app
    app = create_app()
    app.run(debug=True, port=8050)
"""

import dash
import dash_bootstrap_components as dbc

from exohunter import config
from exohunter.dashboard.callbacks import register_callbacks
from exohunter.dashboard.layouts import make_layout
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(pipeline_data: dict | None = None) -> dash.Dash:
    """Create and configure the ExoHunter dashboard application.

    Args:
        pipeline_data: Optional pre-loaded pipeline results to inject
            into the dashboard's data store.  If ``None``, the dashboard
            starts empty and expects data to be loaded interactively.

    Returns:
        A configured ``dash.Dash`` application instance.
    """
    logger.info("Creating ExoHunter dashboard application")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="ExoHunter — Transit Detection Dashboard",
        suppress_callback_exceptions=True,
    )

    # Set the layout
    layout = make_layout()

    # If pipeline data is provided, inject it into the Store component
    if pipeline_data is not None:
        # We need to modify the Store's data after layout creation.
        # Dash's Store component accepts initial data via the `data` prop,
        # but since make_layout() already created it, we wrap the layout
        # in a callback that fires on page load.
        app.layout = layout
        _inject_pipeline_data(app, pipeline_data)
    else:
        app.layout = layout

    # Register all interactive callbacks
    register_callbacks(app)

    logger.info("Dashboard application created successfully")
    return app


def _inject_pipeline_data(app: dash.Dash, data: dict) -> None:
    """Inject pipeline data into the Store component on page load.

    This uses a clientside callback triggered by the URL pathname
    to set the initial data without needing a server roundtrip.
    """
    from dash import Input, Output

    @app.callback(
        Output("pipeline-data", "data"),
        Input("pipeline-data", "id"),  # Fires once on page load
    )
    def load_initial_data(_: str) -> dict:
        return data
