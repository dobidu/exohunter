# ExoHunter

**Concurrent exoplanet transit detection pipeline for NASA TESS data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)]()

---

## What is this?

When a planet passes in front of its host star, the star's brightness
dips by a tiny fraction — typically less than 1%. NASA's
[TESS](https://tess.mit.edu/) space telescope stares at hundreds of
thousands of stars, recording their brightness every 2 minutes, producing
enormous datasets where these dips hide in noise.

**ExoHunter** automates the search. It downloads TESS light curves,
cleans them, runs the Box Least Squares (BLS) algorithm to find periodic
dips, validates candidates against astrophysical criteria, and presents
everything in an interactive dashboard.

The project serves a dual purpose:

- **Scientific tool** — a functional pipeline that can recover known
  exoplanets (tested on TOI-700, a system with a rocky planet in the
  habitable zone).
- **Teaching resource** — a documented, well-structured codebase for a
  study group of Computer Science undergraduates at 
  [UFPB](https://www.ci.ufpb.br/) (Universidade Federal da Paraiba),
  focused on concurrent programming and scientific software development.

---

## How it works

```
  MAST Archive (NASA)
        |
        v
  +--------------------------+
  |  1. INGESTION            |   ThreadPoolExecutor
  |  Download light curves   |   8 concurrent threads (I/O-bound)
  |  from TESS via lightkurve|   Retry + local FITS cache
  +-----------+--------------+
              |
              v
  +--------------------------+
  |  2. PREPROCESSING        |   ProcessPoolExecutor
  |  Remove NaNs & outliers  |   N-1 CPU cores (CPU-bound)
  |  Normalize (median -> 1) |
  |  Flatten (Savitzky-Golay)|
  +-----------+--------------+
              |
              v
  +--------------------------+
  |  3. DETECTION            |   Numba @njit(parallel=True)
  |  BLS periodogram search  |   SIMD-accelerated inner loop
  |  6-criterion validation  |
  |  Transit model fitting   |
  +-----------+--------------+
              |
              v
  +--------------------------+
  |  4. CATALOG              |
  |  Store candidates        |
  |  Cross-match with known  |
  |  planets, export CSV     |
  +-----------+--------------+
              |
              v
  +--------------------------+
  |  5. DASHBOARD            |   Plotly Dash (DARKLY theme)
  |  Sky map (RA/Dec)        |
  |  Interactive light curve |
  |  Phase-folded diagram    |
  |  Candidate table         |
  +--------------------------+
```

### Concurrency model at a glance

| Stage | Bottleneck | Strategy | Why |
|-------|-----------|----------|-----|
| Download | Network latency | `ThreadPoolExecutor` | Threads release the GIL during I/O waits |
| Preprocessing | CPU (SG filter) | `ProcessPoolExecutor` | Separate processes bypass the GIL |
| BLS search | CPU (nested loops) | `numba.prange` | JIT-compiled, SIMD-vectorized parallelism |

---

## Installation

### Requirements

- Python 3.11 or later
- ~2 GB disk for TESS data cache (optional — tests run offline)

### Steps

```bash
# Clone the repository
git clone https://github.com/dobidu/exohunter.git
cd exohunter

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows (PowerShell)

# Install with development dependencies
pip install -e ".[dev]"

# Verify the installation
pytest
```

All 28 tests should pass without internet access.

---

## Quick start

### 1. Demo: detect transits in TOI-700

[TOI-700](https://exoplanets.nasa.gov/exoplanet-catalog/7467/toi-700-d/)
is a nearby M-dwarf with three confirmed planets. Planet **d** orbits in
the habitable zone — the region where liquid water could exist.

```bash
python scripts/demo_single_star.py
```

The script downloads all available TESS sectors for TOI-700, runs the
full pipeline, prints a summary of detected candidates, and saves
interactive HTML plots to `data/output/`.

### 2. Launch the interactive dashboard

```bash
# Synthetic demo data (no internet needed)
python scripts/run_dashboard.py --demo

# Or with real pipeline results
python scripts/run_dashboard.py
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

The dashboard has three linked panels:

| Panel | Shows | Interaction |
|-------|-------|-------------|
| **Sky map** | All processed targets in RA/Dec | Click a star to select it |
| **Light curve** | Time series with transit model overlay | Range slider, model toggle |
| **Phase diagram** | Phase-folded data at the detected period | Binned data + model fit |

Plus a filterable, sortable candidate table with CSV export.

### 3. Run the CLI pipeline

```bash
# Single target by TIC ID
python scripts/run_pipeline.py --tic "TIC 150428135"

# Batch: first 10 targets from TESS sector 1
python scripts/run_pipeline.py --sector 1 --limit 10

# Custom period range (default: 0.5–20 days)
python scripts/run_pipeline.py --tic "TIC 150428135" --min-period 1.0 --max-period 40.0
```

Results are saved to `data/output/candidates.csv`.

---

## Project structure

```
exohunter/
├── config.py                    # Paths, physical constants, default parameters
│
├── ingestion/                   # Layer 1: Data acquisition
│   ├── downloader.py            #   Concurrent TESS downloads (lightkurve + threads)
│   └── cache.py                 #   Local FITS cache to avoid re-downloads
│
├── preprocessing/               # Layer 2: Signal cleaning
│   ├── pipeline.py              #   Orchestrator (ProcessPoolExecutor)
│   ├── clean.py                 #   NaN removal, sigma-clipping
│   ├── normalize.py             #   Median flux → 1.0
│   └── detrend.py               #   Savitzky-Golay flattening
│
├── detection/                   # Layer 3: Transit search
│   ├── bls.py                   #   BLS via lightkurve + Numba from-scratch
│   ├── validator.py             #   6-criterion candidate validation
│   └── model.py                 #   Trapezoidal transit model, phase-folding
│
├── catalog/                     # Layer 4: Results management
│   ├── candidates.py            #   In-memory catalog with filtering
│   ├── crossmatch.py            #   Match against known exoplanets
│   └── export.py                #   CSV and FITS export
│
├── dashboard/                   # Layer 5: Visualization
│   ├── app.py                   #   Dash application factory
│   ├── layouts.py               #   Bootstrap DARKLY layout
│   ├── callbacks.py             #   Interactive callback logic
│   └── figures.py               #   Plotly figure generators
│
└── utils/                       # Shared infrastructure
    ├── logging.py               #   Structured logging (never print())
    ├── timing.py                #   @timing decorator for profiling
    └── parallel.py              #   Thread/process pool wrappers with tqdm
```

### Key files outside the package

| File | Purpose |
|------|---------|
| `scripts/demo_single_star.py` | End-to-end demo on TOI-700 |
| `scripts/run_pipeline.py` | CLI for single-target or batch processing |
| `scripts/run_dashboard.py` | Dashboard server launcher |
| `tests/conftest.py` | Synthetic light curve generators for offline tests |
| `notebooks/01_exploratory.ipynb` | Jupyter notebook for interactive exploration |

---

## Validation criteria

Every BLS detection passes through six tests before being accepted:

| # | Test | Threshold | Rationale |
|---|------|-----------|-----------|
| 1 | **SNR** | >= 7.0 | Community standard for transit detection |
| 2 | **Depth** | 0.01%–5% | Too shallow = noise; too deep = eclipsing binary |
| 3 | **Duration/period** | 0.1%–25% | Must be physically consistent with Kepler's 3rd law |
| 4 | **Transit count** | >= 3 | Minimum for a reliable periodic signal |
| 5 | **V-shape** | <= 0.5 | Box-like = planet; V-shaped = eclipsing binary |
| 6 | **Harmonics** | Not 2:1 or 3:1 | Flags period aliases of stronger signals |

Tests 1–4 are hard requirements. Tests 5–6 produce warnings.

---

## Running tests

```bash
# All tests — runs offline, no network required
pytest

# Verbose output
pytest -v

# With coverage report
pytest --cov=exohunter --cov-report=term-missing

# Single test file
pytest tests/test_bls.py -v
```

The test suite (28 tests) uses synthetic light curves with injected
transits at known periods and depths. It verifies:

- **Preprocessing**: transit signal survives cleaning; outliers removed;
  normalization produces median = 1.0.
- **BLS detection**: correct period recovered within 0.05 days; depth
  within 2x of injected value; Numba implementation agrees with lightkurve.
- **Validation**: each of the six criteria correctly accepts good
  candidates and rejects bad ones.

---

## Technology stack

| Domain | Libraries |
|--------|-----------|
| TESS data access | [lightkurve](https://docs.lightkurve.org/), [astroquery](https://astroquery.readthedocs.io/), [astropy](https://www.astropy.org/) |
| Numerical computing | numpy, scipy, [numba](https://numba.pydata.org/) |
| Visualization | [plotly](https://plotly.com/python/), [dash](https://dash.plotly.com/), dash-bootstrap-components |
| Concurrency | concurrent.futures (`ThreadPoolExecutor`, `ProcessPoolExecutor`), numba `prange` |
| ML (planned) | scikit-learn |
| Testing | pytest, pytest-cov |

---

## Contributing

This project is designed for students. See
[CONTRIBUTING.md](CONTRIBUTING.md) for:

- Setup instructions and development workflow
- Code conventions (type hints, Google-style docstrings, structured logging)
- Commit message format (Conventional Commits)
- Open contribution areas organized by difficulty level (beginner /
  intermediate / advanced)

---

## Roadmap

- [ ] BLS implementation in C with OpenMP for comparison benchmarks
- [ ] ML candidate classification (Random Forest, then CNN on phase curves)
- [ ] Real-time query to the NASA Exoplanet Archive via astroquery TAP
- [ ] Automatic alerts for new candidate detections
- [ ] Batch mode: process an entire TESS sector end-to-end
- [ ] VOTable export for Virtual Observatory interoperability
- [ ] GPU-accelerated BLS using CUDA (via Numba or CuPy)
- [ ] Multi-planet search: iteratively subtract detected transits and re-run BLS

---

## License

[MIT](LICENSE)

## Acknowledgments

- **TESS** — This project uses public data from the Transiting Exoplanet
  Survey Satellite, a NASA Explorer mission led by MIT and operated by
  MIT Lincoln Laboratory.
- **MAST** — Data accessed via the Mikulski Archive for Space Telescopes
  at the Space Telescope Science Institute.
- **lightkurve** — The [lightkurve](https://docs.lightkurve.org/) package
  is developed by the Kepler/K2 and TESS community.

---
