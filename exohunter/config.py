"""Global configuration for the ExoHunter pipeline.

Centralizes all paths, physical constants, and default parameters
so that no module needs to hardcode values. All paths are relative
to the project root and constructed with pathlib.Path.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
CATALOG_DIR: Path = DATA_DIR / "catalogs"
OUTPUT_DIR: Path = DATA_DIR / "output"

# Ensure data directories exist at import time
for _dir in (CACHE_DIR, CATALOG_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TESS mission parameters
# ---------------------------------------------------------------------------

# A single TESS sector observes for approximately 27.4 days
TESS_SECTOR_DURATION_DAYS: float = 27.4

# TESS short-cadence exposure time in days (~2 minutes)
TESS_SHORT_CADENCE_DAYS: float = 2.0 / (60.0 * 24.0)

# ---------------------------------------------------------------------------
# Ingestion defaults
# ---------------------------------------------------------------------------

# Maximum number of concurrent download threads (I/O-bound)
DEFAULT_DOWNLOAD_WORKERS: int = 8

# HTTP timeout for a single lightkurve query/download (seconds)
DOWNLOAD_TIMEOUT_SECONDS: int = 120

# Number of retries on transient network errors
DOWNLOAD_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------

# Sigma clipping threshold for outlier removal
OUTLIER_SIGMA: float = 5.0

# Window length for the Savitzky-Golay flatten filter (must be odd)
FLATTEN_WINDOW_LENGTH: int = 1001

# Transit duration (hours) used when estimating CDPP
CDPP_TRANSIT_DURATION_HOURS: float = 13.0

# ---------------------------------------------------------------------------
# BLS detection defaults
# ---------------------------------------------------------------------------

# Period search grid (days)
BLS_MIN_PERIOD_DAYS: float = 0.5
BLS_MAX_PERIOD_DAYS: float = 20.0
BLS_NUM_PERIODS: int = 10_000

# Duration grid (hours) — typical transit durations for hot Jupiters to
# super-Earths around Sun-like stars
BLS_DURATIONS_HOURS: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Frequency factor passed to lightkurve's BLS periodogram
BLS_FREQUENCY_FACTOR: int = 500

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

# Minimum signal-to-noise ratio for a candidate to be considered real
MIN_SNR: float = 7.0

# Transit depth bounds (fractional flux decrease)
MIN_DEPTH: float = 1e-4   # 0.01 %  — shallower is likely noise
MAX_DEPTH: float = 0.05   # 5 %     — deeper is likely an eclipsing binary

# Minimum number of observed transits
MIN_TRANSITS: int = 3

# V-shape metric threshold (0 = perfect box, 1 = perfect V)
# Candidates above this are flagged as possible eclipsing binaries
MAX_V_SHAPE: float = 0.5

# ---------------------------------------------------------------------------
# Dashboard defaults
# ---------------------------------------------------------------------------

DASHBOARD_HOST: str = "127.0.0.1"
DASHBOARD_PORT: int = 8050
DASHBOARD_DEBUG: bool = True

# Data source for the dashboard when launched without explicit flags.
# Options:
#   "demo"  — load synthetic TOI-700 demonstration data (default)
#   "csv"   — load candidates from data/output/candidates.csv
#   "none"  — start with an empty dashboard
DASHBOARD_DATA_SOURCE: str = "demo"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
