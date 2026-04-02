# ExoHunter

**Concurrent exoplanet transit detection pipeline for NASA TESS data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

ExoHunter is a complete pipeline for detecting exoplanet transits in data
from the [TESS](https://tess.mit.edu/) space telescope. It downloads light
curves from MAST, preprocesses them with concurrent workers, runs the
Box Least Squares (BLS) algorithm to find periodic dips, validates
candidates against astrophysical criteria, and presents results in an
interactive Plotly Dash dashboard.

This project is both a **scientific tool** and a **pedagogical resource**
for a study group of Computer Science undergraduates at UFPB (Universidade
Federal da Paraiba), focused on software development for astronomical
applications.

## Architecture

```
                    ┌──────────────┐
                    │  MAST / TESS │  (NASA archive)
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   ingestion/            │  ThreadPoolExecutor
              │   Concurrent download   │  (I/O-bound → threads)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   preprocessing/        │  ProcessPoolExecutor
              │   Clean, normalize,     │  (CPU-bound → processes)
              │   detrend               │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   detection/            │  Numba @njit(parallel)
              │   BLS + validation      │  (CPU-bound → SIMD)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   catalog/              │
              │   Cross-match, export   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   dashboard/            │  Plotly Dash
              │   Interactive viz       │  (DARKLY theme)
              └─────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/exohunter-ufpb/exohunter.git
cd exohunter

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Demo: Detect transits in TOI-700

TOI-700 is a nearby M-dwarf star with three confirmed planets, including
TOI-700 d — a rocky planet in the habitable zone.

```bash
python scripts/demo_single_star.py
```

This will download the TESS data, run the full pipeline, and generate
output plots in `data/output/`.

### Run the interactive dashboard

```bash
# With synthetic demo data (no download needed)
python scripts/run_dashboard.py --demo

# With real pipeline results
python scripts/run_dashboard.py
```

Open your browser at [http://localhost:8050](http://localhost:8050).

### Run the CLI pipeline

```bash
# Single target
python scripts/run_pipeline.py --tic "TIC 150428135"

# Process targets from a TESS sector
python scripts/run_pipeline.py --sector 1 --limit 10
```

## Project Structure

| Module | Responsibility | Concurrency |
|--------|---------------|-------------|
| `exohunter/ingestion/` | Download TESS light curves from MAST | `ThreadPoolExecutor` (I/O-bound) |
| `exohunter/preprocessing/` | Clean, normalize, detrend light curves | `ProcessPoolExecutor` (CPU-bound) |
| `exohunter/detection/` | BLS periodogram, transit validation | `numba.njit(parallel=True)` |
| `exohunter/catalog/` | Candidate catalog, cross-matching, export | — |
| `exohunter/dashboard/` | Interactive Plotly Dash visualization | — |
| `exohunter/utils/` | Logging, timing decorators, parallel helpers | — |

## Running Tests

```bash
# All tests (offline, no network needed)
pytest

# With coverage
pytest --cov=exohunter

# Specific module
pytest tests/test_bls.py -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for a guide designed for students
and new contributors.

## Roadmap

- [ ] BLS implementation in C with OpenMP
- [ ] ML classification (Random Forest → CNN)
- [ ] Automatic alerts for new candidates
- [ ] Integration with ExoFOP for validation
- [ ] Batch mode: process entire TESS sectors
- [ ] VOTable export format
- [ ] GPU-accelerated BLS (CUDA)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Credits

- **TESS data**: This project uses data from the Transiting Exoplanet
  Survey Satellite (TESS), a NASA Explorer mission managed by MIT and
  operated by MIT Lincoln Laboratory.
- **MAST archive**: Data accessed via the Mikulski Archive for Space
  Telescopes (MAST) at STScI.
- **lightkurve**: The [lightkurve](https://docs.lightkurve.org/) package
  is developed by the Kepler/K2 and TESS community.
