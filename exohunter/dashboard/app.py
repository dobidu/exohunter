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

    # Pass pipeline data directly to the layout so the Store component
    # is initialised with the data at construction time.  This is simpler
    # and more reliable than injecting data via a callback.
    app.layout = make_layout(pipeline_data=pipeline_data)

    # Register all interactive callbacks
    register_callbacks(app)

    logger.info("Dashboard application created successfully")
    return app
