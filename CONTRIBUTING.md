# Contributing to ExoHunter

Welcome! This guide is designed for **undergraduates** joining the study
group. No prior astronomy knowledge is needed — just Python and curiosity.

## Getting Started

### 1. Fork and clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/exohunter.git
cd exohunter
```

### 2. Set up your environment

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 3. Verify everything works

```bash
pytest
```

All tests should pass without internet access.

## Development Workflow

### Branch naming

Create a branch from `main` for each feature or fix:

```bash
git checkout -b feat/my-feature    # New feature
git checkout -b fix/bug-name       # Bug fix
git checkout -b docs/topic         # Documentation
git checkout -b refactor/module    # Refactoring
```

### Code conventions

- **Python 3.11+** — use modern syntax (type hints, `match`, `|` unions)
- **Type hints** on all public functions
- **Docstrings** in Google style on all public modules and functions
- **Logging** via `from exohunter.utils.logging import get_logger` — never `print()`
- **Variable names** in English; comments and docstrings in English
- Prefer **explicit and readable** over clever and terse

Example:

```python
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_transit_depth(
    flux_in_transit: np.ndarray,
    flux_out_of_transit: np.ndarray,
) -> float:
    """Calculate the fractional transit depth.

    The transit depth is defined as the difference between the
    out-of-transit and in-transit median flux, divided by the
    out-of-transit median.

    Args:
        flux_in_transit: Flux measurements during transit.
        flux_out_of_transit: Flux measurements outside transit.

    Returns:
        Fractional depth (e.g. 0.01 means the star dims by 1%).
    """
    median_out = np.median(flux_out_of_transit)
    median_in = np.median(flux_in_transit)
    depth = (median_out - median_in) / median_out

    logger.debug("Transit depth: %.6f (%.4f%%)", depth, depth * 100)
    return depth
```

### Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add VOTable export to catalog module
fix: handle NaN values in BLS period grid
docs: add installation instructions for Windows
refactor: extract phase-folding into separate function
test: add edge case tests for validator depth check
```

### Running tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_bls.py

# With verbose output
pytest -v

# With coverage report
pytest --cov=exohunter --cov-report=html
```

Tests must run **offline** (no network access). Use synthetic data
from `tests/conftest.py`.

### Pull request process

1. Push your branch: `git push origin feat/my-feature`
2. Open a Pull Request on GitHub against `main`
3. Describe what you changed and why
4. Wait for code review — the professor will review your PR
5. Address feedback and push fixes
6. Once approved, your PR will be merged

## Areas Open for Contribution

Each of these is a self-contained project that a student can take on:

### Beginner

- Add more validation criteria to `detection/validator.py`
- Improve error messages and logging throughout the codebase
- Add more synthetic test cases to `tests/`
- Write a Jupyter notebook tutorial for a specific module

### Intermediate

- Implement VOTable export in `catalog/export.py` using `astropy.io.votable`
- Add a stellar classification panel to the dashboard
- Implement a CLI progress bar for batch processing
- Create a configuration file parser (YAML/TOML) for pipeline parameters

### Advanced

- Implement BLS in C with Python bindings (ctypes or cffi)
- Add a Random Forest classifier for candidate triage in `detection/`
- Build a real-time query to NASA Exoplanet Archive in `catalog/crossmatch.py`
- Implement GPU-accelerated BLS using CUDA (via Numba or CuPy)
- Add OpenMP parallelism to the C BLS implementation

## Questions?

Open an issue on GitHub or ask in the study group chat.
