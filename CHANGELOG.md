# Changelog

All notable changes to ExoHunter are documented in this file.

## [1.0.0] — 2026-04-04

### First public release

ExoHunter is a concurrent exoplanet transit detection pipeline for
NASA TESS data, built as a teaching resource for the Computer Science
study group at UFPB (Universidade Federal da Paraiba).

### Pipeline

- **Ingestion**: concurrent TESS light curve download via lightkurve
  with ThreadPoolExecutor (8 threads), exponential backoff retry,
  and local FITS cache (astropy Table I/O for reliable roundtrip)
- **Preprocessing**: clean (NaN removal, 5-sigma clipping), normalize,
  Savitzky-Golay detrending via ProcessPoolExecutor (N-1 cores)
- **BLS detection**: two implementations — lightkurve/astropy (C) and
  Numba prefix-sum (5.8x faster); iterative multi-planet search via
  transit model subtraction (up to 5 planets per star)
- **Validation**: 6-criterion astrophysical validator (SNR, depth,
  duration, transit count, V-shape, harmonics)
- **Cross-matching**: 4-tier classification (NEW_CANDIDATE, KNOWN_MATCH,
  KNOWN_TOI, HARMONIC) against ExoFOP-TESS TOI catalog via NASA
  Exoplanet Archive TAP API with 48-hour auto-refresh and 3-tier
  fallback (TAP → ExoFOP HTTP → local CSV → built-in table)
- **Scoring**: priority score = SNR x v_shape_factor x depth_factor
- **Export**: CSV, FITS, and VOTable (with IVOA UCD metadata)
- **Alerts**: automatic file + webhook dispatch for NEW_CANDIDATE
  detections with SNR >= 7

### ML Classification

- **Random Forest** (scikit-learn): 10 tabular features, trained on
  Kepler KOI cumulative table (~7,600 labeled examples), 3 classes
  (planet, eclipsing_binary, false_positive)
- **1D CNN** (PyTorch): 201-bin phase-folded light curve input,
  Conv1d(1→16→32→64) → FC(64→32→3), trained on synthetic phase
  curves generated from Kepler catalog parameters
- Both classifiers are opt-in via `--classify` (RF) or
  `--classify-cnn` flags

### Batch Processing

- `run_batch.py`: full sector processing with magnitude filtering
  (TIC catalog query), multi-sector stitching (`--multi-sector`),
  single-sector restriction (`--single-sector`), multi-planet
  search (`--multi-planet`), ML classification (`--classify` /
  `--classify-cnn`)
- `inspect_candidate.py`: 4-panel PNG diagnostic report (light curve,
  phase-fold with trapezoidal fit, BLS periodogram, odd-even EB check)

### Dashboard

- Plotly Dash with Bootstrap DARKLY theme
- 5 visualization panels: sky map, light curve with transit windows,
  phase-folded diagram, BLS periodogram, odd-even transit comparison
- Data source selector with auto-scan of batch results
- New Candidates highlight panel (top 10 uncatalogued by score)
- Classification filter (4-tier) + period/SNR/status filters
- Multi-planet candidate selector dropdown
- Data Overview section: cache stats, ML model status, batch results
  index, reports gallery with modal viewer, alerts feed timeline
- ML Class and P(planet) columns in candidate table

### Notebooks

- `00_lecture_introduction.ipynb`: theory + code lecture with 6
  interactive animations (discovery timeline, transit depth calculator,
  preprocessing steps, BLS phase-fold, validation flowchart, feature
  importances)
- `01_exploratory.ipynb`: guided TOI-700 walkthrough with transit
  geometry animation, phase-fold sweep, and planet subtraction sequence
- `02_student_exercises.ipynb`: L 98-59 analysis mission with 40 TODOs,
  injection-recovery experiment, and graded rubric

### Documentation

- `README.md`: project overview, architecture, quick start, validation
  criteria, technology stack, roadmap
- `THEORY.md`: scientific background for CS students (transits, TESS,
  BLS algorithm, validation, TOI-700 system)
- `METHODOLOGY.md`: implementation details (all pipeline stages,
  algorithms, data structures, configuration reference)
- `ML_GUIDE.md`: ML classification walkthrough (datasets, features,
  RF training, CNN architecture, comparison)
- `DASHBOARD.md`: dashboard user guide (panels, filters, interpretation,
  investigation workflow)
- `CONTRIBUTING.md`: contribution guide for students

### Testing

- 192 tests across 12 test files, all running offline (no network)
- Covers: BLS (lightkurve + Numba + iterative), cache roundtrip,
  catalog (scoring, ranking, crossmatch, staleness), classification
  (RF + CNN), dashboard (figures, layout, demo data), export
  (CSV/FITS/VOTable), alerts, data overview, transit model,
  preprocessing, validation, utilities
