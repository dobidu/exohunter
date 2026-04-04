"""ExoHunter — Concurrent exoplanet transit detection pipeline.

A pipeline for detecting exoplanet transits in NASA TESS data,
featuring concurrent data processing, BLS periodogram analysis,
and an interactive Plotly Dash dashboard for visualization.
"""

__version__ = "1.0.0"
__author__ = "ExoHunter Team — UFPB"

# Suppress the astropy UnitsWarning for 'btjd' (Barycentric TESS Julian Date).
# BTJD is the standard time system for TESS data but is not part of the FITS
# unit standard.  lightkurve uses it internally and this warning is harmless.
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", message=".*did not parse as fits unit.*", category=AstropyWarning)
