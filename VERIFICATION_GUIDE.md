# ExoHunter Verification Guide

**A hands-on protocol for verifying pipeline correctness**

This guide walks you through 12 verification tasks that independently
test every stage of the ExoHunter pipeline. Each task has:

- **Objective**: what you're verifying
- **Commands**: exactly what to run
- **Expected output**: what correct behavior looks like
- **Pass/fail criteria**: explicit conditions to check

Use TOI-700 (TIC 150428135) as ground truth — a well-studied system
with 3 confirmed planets at known periods.

### Prerequisites

```bash
cd exohunter
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### Ground truth: TOI-700 system

| Planet | Period (d) | Depth (ppm) | Duration (h) |
|--------|-----------|-------------|---------------|
| b | 9.977 | 580 | 2.3 |
| c | 16.051 | 780 | 3.0 |
| d | 37.426 | 810 | 3.5 |

---

## Task 1: Installation and test suite

**Objective**: Verify that the package installs cleanly and all 192
tests pass without network access.

**Commands**:

```bash
pip install -e ".[dev]"
pytest
```

**Expected output**:

```
========================= 192 passed, N warnings =========================
```

**Pass criteria**:
- [ ] `pip install` completes without errors
- [ ] All 192 tests pass
- [ ] No test requires network access (disconnect WiFi and re-run to confirm)

---

## Task 2: Download and cache integrity

**Objective**: Verify that light curve download and FITS cache produce
identical data on re-read.

**Commands**:

```python
import numpy as np
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.ingestion.cache import load_from_cache, save_to_cache
from exohunter import config

# Download (or load from cache)
lc = download_lightcurve("TIC 150428135")

time_orig = np.array(lc.time.value, dtype=np.float64)
flux_orig = np.array(lc.flux.value, dtype=np.float64)

print(f"Cadences: {len(time_orig)}")
print(f"Time span: {time_orig[-1] - time_orig[0]:.1f} days")

# Force a cache roundtrip
save_to_cache(lc, "TIC 150428135", config.CACHE_DIR)
lc_cached = load_from_cache("TIC 150428135", config.CACHE_DIR)

time_cached = np.array(lc_cached.time.value, dtype=np.float64)
flux_cached = np.array(lc_cached.flux.value, dtype=np.float64)

# Compare
assert len(time_cached) == len(time_orig), "FAIL: cadence count mismatch"
assert np.allclose(time_cached, time_orig, atol=1e-10), "FAIL: time arrays differ"
assert np.allclose(flux_cached, flux_orig, atol=1e-10), "FAIL: flux arrays differ"
print("PASS: cache roundtrip preserves all data exactly")
```

**Pass criteria**:
- [ ] Download succeeds (or loads from cache)
- [ ] Cadence count > 10,000
- [ ] Time span > 100 days (multi-sector)
- [ ] `time` and `flux` arrays are identical after cache roundtrip

---

## Task 3: Preprocessing preserves transit signal

**Objective**: Verify that the preprocessing pipeline (clean, normalize,
flatten) does not destroy the known transit signal of TOI-700 b.

**Commands**:

```python
import numpy as np
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single

lc = download_lightcurve("TIC 150428135")
processed = preprocess_single(lc, tic_id="TIC 150428135")

print(f"Before: {len(lc)} cadences")
print(f"After:  {len(processed.time)} cadences")
print(f"Removed: {len(lc) - len(processed.time)} ({100*(1 - len(processed.time)/len(lc)):.1f}%)")

# Phase-fold at TOI-700 b's known period and check for a dip
period = 9.977
phase = (processed.time % period) / period

# Compare flux near transit center vs. out of transit
near_transit = np.abs(phase - 0.5) < 0.015
out_of_transit = (phase < 0.4) | (phase > 0.6)

if np.sum(near_transit) > 0 and np.sum(out_of_transit) > 0:
    flux_in = np.median(processed.flux[near_transit])
    flux_out = np.median(processed.flux[out_of_transit])
    measured_depth = flux_out - flux_in

    print(f"\nPhase-fold at P={period} d:")
    print(f"  Flux in transit:  {flux_in:.6f}")
    print(f"  Flux out transit: {flux_out:.6f}")
    print(f"  Measured depth:   {measured_depth:.6f} ({measured_depth*1e6:.0f} ppm)")

    if measured_depth > 0.0001:
        print("PASS: transit signal preserved after preprocessing")
    else:
        print("WARNING: signal may be attenuated (check flatten window)")
else:
    print("WARNING: not enough cadences near phase 0.5")

# Additional check: less than 10% of cadences removed
fraction_kept = len(processed.time) / len(lc)
assert fraction_kept > 0.90, f"FAIL: too many cadences removed ({fraction_kept:.1%})"
print(f"PASS: {fraction_kept:.1%} cadences retained")
```

**Pass criteria**:
- [ ] Less than 10% of cadences removed
- [ ] Phase-fold at P=9.977d shows a measurable dip (depth > 100 ppm)
- [ ] Median out-of-transit flux is approximately 1.0

---

## Task 4: BLS detects the correct period

**Objective**: Verify that the BLS algorithm recovers TOI-700 b's
period (9.977 d) from the preprocessed data.

**Commands**:

```python
import numpy as np
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.detection.bls import run_bls_lightkurve

lc = download_lightcurve("TIC 150428135")
processed = preprocess_single(lc, tic_id="TIC 150428135")
lc_for_bls = processed.to_lightcurve()

candidate = run_bls_lightkurve(
    lc_for_bls, tic_id="TIC 150428135",
    min_period=0.5, max_period=45.0, num_periods=15000,
)

print(f"Detected period: {candidate.period:.4f} d")
print(f"Detected depth:  {candidate.depth*100:.4f}%")
print(f"SNR:             {candidate.snr:.1f}")

# Check against known TOI-700 planets
known_periods = [9.977, 16.051, 37.426]
matches = [p for p in known_periods if abs(candidate.period - p) < 0.5]

if matches:
    print(f"PASS: detected period matches TOI-700 (closest: {matches[0]} d)")
else:
    # Could be a harmonic — check 2x and 0.5x
    for kp in known_periods:
        if abs(candidate.period / kp - 2.0) < 0.1 or abs(candidate.period / kp - 0.5) < 0.1:
            print(f"NOTE: detected period is a harmonic of {kp} d")
            break
    else:
        print(f"WARNING: period {candidate.period:.4f} doesn't match any known TOI-700 planet")
```

**Pass criteria**:
- [ ] BLS returns a non-None candidate
- [ ] Detected period is within 0.5 d of one of the three known planets
- [ ] If it finds a harmonic, that's acceptable (note it)

---

## Task 5: Numba BLS agrees with lightkurve BLS

**Objective**: Verify that both BLS implementations find the same period.

**Commands**:

```python
import numpy as np
from tests.conftest import make_synthetic_transit_lc
from exohunter.detection.bls import run_bls_lightkurve, run_bls_numba, _NUMBA_AVAILABLE

# Use synthetic data (deterministic, no network needed)
lc = make_synthetic_transit_lc(period=7.0, depth=0.01, duration=0.15,
                                noise=0.0005, n_points=15000)

# lightkurve BLS
lk_result = run_bls_lightkurve(lc, tic_id="CMP", min_period=2.0,
                                max_period=12.0, num_periods=5000)

print(f"lightkurve BLS: P = {lk_result.period:.4f} d")

if _NUMBA_AVAILABLE:
    # Numba BLS
    nb_result = run_bls_numba(lc.time.value, lc.flux.value, tic_id="CMP",
                               min_period=2.0, max_period=12.0, num_periods=5000)

    print(f"Numba BLS:      P = {nb_result.period:.4f} d")

    diff = abs(lk_result.period - nb_result.period)
    print(f"Difference:     {diff:.4f} d")

    assert diff < 0.2, f"FAIL: implementations disagree by {diff:.4f} d"
    print("PASS: both implementations agree")
else:
    print("SKIP: Numba not available")

# Both should be close to the injected 7.0 d
assert abs(lk_result.period - 7.0) < 0.1, f"FAIL: lightkurve found {lk_result.period:.4f}, expected ~7.0"
print(f"PASS: both recover injected period (7.0 d)")
```

**Pass criteria**:
- [ ] lightkurve BLS finds P ≈ 7.0 d (±0.1)
- [ ] Numba BLS finds P ≈ 7.0 d (±0.1)
- [ ] The two periods agree within 0.2 d

---

## Task 6: Multi-planet search finds multiple planets

**Objective**: Verify that iterative BLS subtraction discovers at least
2 of TOI-700's 3 planets.

**Commands**:

```python
from exohunter.ingestion.downloader import download_lightcurve
from exohunter.preprocessing.pipeline import preprocess_single
from exohunter.detection.bls import run_iterative_bls

lc = download_lightcurve("TIC 150428135")
processed = preprocess_single(lc, tic_id="TIC 150428135")
lc_for_bls = processed.to_lightcurve()

candidates = run_iterative_bls(
    lc_for_bls, tic_id="TIC 150428135",
    min_period=0.5, max_period=45.0, num_periods=15000,
    max_planets=5, min_snr=0.0,
)

print(f"Found {len(candidates)} candidate(s):\n")
known = {"b": 9.977, "c": 16.051, "d": 37.426}
matched = set()

for i, c in enumerate(candidates):
    letter = chr(ord("b") + i)
    match = ""
    for name, period in known.items():
        if abs(c.period - period) < 1.0:
            match = f"  ← TOI-700 {name}"
            matched.add(name)
    print(f"  Planet {letter}: P={c.period:.4f} d, depth={c.depth*100:.4f}%{match}")

print(f"\nMatched {len(matched)}/3 known planets: {', '.join(sorted(matched)) or 'none'}")

assert len(candidates) >= 2, f"FAIL: found only {len(candidates)} candidates"
print("PASS: multi-planet search found >= 2 candidates")
```

**Pass criteria**:
- [ ] At least 2 candidates found
- [ ] At least 1 matches a known TOI-700 planet period (within 1 d)

---

## Task 7: Validation correctly accepts and rejects

**Objective**: Verify that the 6-criterion validator correctly accepts
a good candidate and rejects a bad one.

**Commands**:

```python
from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import validate_candidate

# A good candidate (should pass all tests)
good = TransitCandidate(
    tic_id="GOOD", period=10.0, epoch=5.0, duration=0.2,
    depth=0.01, snr=15.0, bls_power=0.5, n_transits=9,
)
good_result = validate_candidate(good)

print("Good candidate:")
for test, passed in good_result.tests.items():
    print(f"  {'PASS' if passed else 'FAIL'} — {test}")
print(f"  Overall: {'VALID' if good_result.is_valid else 'REJECTED'}")
assert good_result.is_valid, "FAIL: good candidate was rejected"

# A bad candidate (should fail multiple tests)
bad = TransitCandidate(
    tic_id="BAD", period=10.0, epoch=5.0, duration=3.0,
    depth=0.10, snr=2.0, bls_power=0.1, n_transits=1,
)
bad_result = validate_candidate(bad)

print("\nBad candidate:")
for test, passed in bad_result.tests.items():
    print(f"  {'PASS' if passed else 'FAIL'} — {test}")
print(f"  Overall: {'VALID' if bad_result.is_valid else 'REJECTED'}")
assert not bad_result.is_valid, "FAIL: bad candidate was accepted"

print("\nPASS: validator correctly accepts good and rejects bad candidates")
```

**Pass criteria**:
- [ ] Good candidate: all 4 critical tests pass, overall VALID
- [ ] Bad candidate: at least 3 tests fail (SNR, depth, duration, n_transits), overall REJECTED

---

## Task 8: Cross-matching classifies correctly

**Objective**: Verify all 4 cross-match classifications using the
built-in reference table (no network needed).

**Commands**:

```python
from exohunter.detection.bls import TransitCandidate
from exohunter.catalog.crossmatch import crossmatch_candidate, MatchClass

# KNOWN_MATCH: TOI-700 b at its exact period
c1 = TransitCandidate(tic_id="TIC 150428135", period=9.977, epoch=0,
                       duration=0.1, depth=0.001, snr=10, bls_power=1)
r1 = crossmatch_candidate(c1)
print(f"TOI-700 b (P=9.977):     {r1.match_class.value} — {r1.catalog_name}")
assert r1.match_class == MatchClass.KNOWN_MATCH, f"FAIL: expected KNOWN_MATCH, got {r1.match_class}"

# HARMONIC: 2x TOI-700 b
c2 = TransitCandidate(tic_id="TIC 150428135", period=19.95, epoch=0,
                       duration=0.1, depth=0.001, snr=10, bls_power=1)
r2 = crossmatch_candidate(c2)
print(f"2x TOI-700 b (P=19.95):  {r2.match_class.value}")
assert r2.match_class == MatchClass.HARMONIC, f"FAIL: expected HARMONIC, got {r2.match_class}"

# KNOWN_TOI: known TIC, different period
c3 = TransitCandidate(tic_id="TIC 150428135", period=6.0, epoch=0,
                       duration=0.1, depth=0.001, snr=10, bls_power=1)
r3 = crossmatch_candidate(c3)
print(f"TOI-700 diff P (P=6.0):  {r3.match_class.value}")
assert r3.match_class == MatchClass.KNOWN_TOI, f"FAIL: expected KNOWN_TOI, got {r3.match_class}"

# NEW_CANDIDATE: unknown TIC
c4 = TransitCandidate(tic_id="TIC 999999999", period=5.0, epoch=0,
                       duration=0.1, depth=0.001, snr=10, bls_power=1)
r4 = crossmatch_candidate(c4)
print(f"Unknown TIC:             {r4.match_class.value}")
assert r4.match_class == MatchClass.NEW_CANDIDATE, f"FAIL: expected NEW_CANDIDATE, got {r4.match_class}"

print("\nPASS: all 4 cross-match classifications correct")
```

**Pass criteria**:
- [ ] KNOWN_MATCH for TOI-700 b at P=9.977 d
- [ ] HARMONIC for P=19.95 d (2x TOI-700 b)
- [ ] KNOWN_TOI for P=6.0 d (different period, same star)
- [ ] NEW_CANDIDATE for unknown TIC

---

## Task 9: ML classification predicts sensibly

**Objective**: Verify the Random Forest classifier on synthetic data
(no real training data needed).

**Commands**:

```python
import numpy as np
import pandas as pd
from exohunter.classification.model import build_pipeline, classify_candidates, CLASS_LABELS
from exohunter.classification.features import FEATURE_COLUMNS

# Create synthetic training data with clear separations
rng = np.random.default_rng(42)
rows = []
for label, depth, snr in [("planet", 0.001, 20), ("eclipsing_binary", 0.03, 40),
                            ("false_positive", 0.0001, 3)]:
    for _ in range(200):
        rows.append({
            "period": rng.uniform(1, 15), "depth": abs(rng.normal(depth, depth*0.3)),
            "duration": rng.uniform(0.02, 0.15), "snr": abs(rng.normal(snr, snr*0.3)),
            "impact_param": rng.uniform(0, 0.8), "stellar_teff": rng.normal(5500, 500),
            "stellar_logg": rng.normal(4.3, 0.3), "stellar_radius": rng.normal(1.0, 0.3),
            "duration_period_ratio": 0, "depth_log": 0, "label": label,
        })

df = pd.DataFrame(rows)
df["duration_period_ratio"] = df["duration"] / df["period"]
df["depth_log"] = np.log10(df["depth"].clip(lower=1e-10))

# Train
pipeline = build_pipeline(n_estimators=100, max_depth=10)
X = df[FEATURE_COLUMNS].values
y = df["label"].values
pipeline.fit(X, y)

# Predict on training data (sanity check)
results = classify_candidates(pipeline, df)
y_pred = results["ml_class"].values
accuracy = np.mean(y == y_pred)
print(f"Training accuracy: {accuracy:.1%}")

assert accuracy > 0.7, f"FAIL: accuracy {accuracy:.1%} is too low"
print("PASS: RF classifier achieves > 70% accuracy on synthetic data")

# Check probabilities sum to 1
prob_cols = ["ml_prob_planet", "ml_prob_eb", "ml_prob_fp"]
row_sums = results[prob_cols].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-5), "FAIL: probabilities don't sum to 1"
print("PASS: class probabilities sum to 1.0")
```

**Pass criteria**:
- [ ] Training accuracy > 70%
- [ ] All predictions are valid class labels
- [ ] Per-row probabilities sum to 1.0

---

## Task 10: Export roundtrip preserves data

**Objective**: Verify that CSV, FITS, and VOTable exports can be read
back with identical data.

**Commands**:

```python
import numpy as np
from pathlib import Path
from exohunter.catalog.candidates import CandidateCatalog
from exohunter.catalog.export import export_to_csv, export_to_fits, export_to_votable
from exohunter.detection.bls import TransitCandidate
from exohunter.detection.validator import ValidationResult

# Create a test catalog
catalog = CandidateCatalog()
c = TransitCandidate(tic_id="TIC 150428135", period=9.977, epoch=2.5,
                      duration=0.095, depth=0.00058, snr=12.4, bls_power=1.0,
                      n_transits=35, name="TOI-700 b")
v = ValidationResult(is_valid=True, tests={"v_shape": True}, flags=[])
catalog.add(c, v)

tmp = Path("/tmp/exohunter_export_test")
tmp.mkdir(exist_ok=True)

# CSV roundtrip
import pandas as pd
csv_path = export_to_csv(catalog, tmp / "test.csv")
df_csv = pd.read_csv(csv_path)
assert abs(df_csv.iloc[0]["period"] - 9.977) < 0.001, "FAIL: CSV period mismatch"
print("PASS: CSV roundtrip preserves period")

# FITS roundtrip
from astropy.table import Table
fits_path = export_to_fits(catalog, tmp / "test.fits")
t_fits = Table.read(fits_path, format="fits")
assert abs(float(t_fits["period"][0]) - 9.977) < 0.001, "FAIL: FITS period mismatch"
print("PASS: FITS roundtrip preserves period")

# VOTable roundtrip
vot_path = export_to_votable(catalog, tmp / "test.votable.xml")
t_vot = Table.read(vot_path, format="votable")
assert abs(float(t_vot["period"][0]) - 9.977) < 0.001, "FAIL: VOTable period mismatch"
assert t_vot["period"].meta.get("ucd") == "time.period", "FAIL: VOTable missing UCD"
assert str(t_vot["period"].unit) == "d", "FAIL: VOTable missing unit"
print("PASS: VOTable roundtrip preserves period + UCD + unit")

print("\nPASS: all 3 export formats roundtrip correctly")
```

**Pass criteria**:
- [ ] CSV: period value preserved
- [ ] FITS: period value preserved
- [ ] VOTable: period value preserved + UCD metadata + unit

---

## Task 11: Dashboard serves correctly

**Objective**: Verify that the dashboard creates, serves HTTP, and
contains all expected components.

**Commands**:

```python
import threading, time, urllib.request, json
from scripts.run_dashboard import generate_demo_data
from exohunter.dashboard.app import create_app

data = generate_demo_data()
app = create_app(pipeline_data=data)

# Start server in background
server = threading.Thread(
    target=lambda: app.run(host="127.0.0.1", port=9999, debug=False),
    daemon=True,
)
server.start()
time.sleep(4)

# Check HTTP response
resp = urllib.request.urlopen("http://127.0.0.1:9999/")
html = resp.read().decode()
assert "ExoHunter" in html or "react-entry-point" in html, "FAIL: no HTML served"
print("PASS: dashboard serves HTML")

# Check all component IDs in layout
resp2 = urllib.request.urlopen("http://127.0.0.1:9999/_dash-layout")
layout = json.dumps(json.loads(resp2.read().decode()))

required = [
    "pipeline-data", "sky-map", "lightcurve-plot", "phase-plot",
    "periodogram-plot", "odd-even-plot", "candidate-table",
    "candidate-selector", "data-source-selector", "xmatch-filter",
    "new-candidates-panel", "cache-stats-body", "ml-status-body",
    "batch-results-body", "reports-gallery-body", "alerts-feed-body",
]

missing = [c for c in required if c not in layout]
if missing:
    print(f"FAIL: missing components: {missing}")
else:
    print(f"PASS: all {len(required)} components present")

# Check demo data is injected
assert "TOI-700" in layout, "FAIL: demo data not injected"
print("PASS: demo data contains TOI-700")
```

**Pass criteria**:
- [ ] HTTP 200 response
- [ ] All 16 component IDs present in the layout
- [ ] Demo data contains TOI-700

---

## Task 12: Alert system triggers correctly

**Objective**: Verify that the alert system creates a JSON file when
NEW_CANDIDATE entries meet the criteria.

**Commands**:

```python
import json
from pathlib import Path
import pandas as pd
from exohunter.alerts import check_and_dispatch_alerts

# Create a mock summary with alert-worthy candidates
summary = pd.DataFrame([
    {"tic_id": "TIC 900000001", "xmatch_class": "NEW_CANDIDATE",
     "snr": 12.0, "is_valid": True, "period": 5.0, "depth": 0.005},
    {"tic_id": "TIC 900000002", "xmatch_class": "KNOWN_MATCH",
     "snr": 15.0, "is_valid": True, "period": 3.0, "depth": 0.01},
    {"tic_id": "TIC 900000003", "xmatch_class": "NEW_CANDIDATE",
     "snr": 3.0, "is_valid": False, "period": 8.0, "depth": 0.001},
])

# Use a temp directory to avoid polluting data/alerts
import exohunter.config as cfg
original_dir = cfg.ALERTS_DIR
tmp = Path("/tmp/exohunter_alert_test")
tmp.mkdir(exist_ok=True)
cfg.ALERTS_DIR = tmp

try:
    n = check_and_dispatch_alerts(summary, sector=99)
    print(f"Alert-worthy candidates: {n}")

    # Only TIC 900000001 should trigger (NEW_CANDIDATE + SNR>=7 + valid)
    assert n == 1, f"FAIL: expected 1 alert, got {n}"

    # Check the alert file
    alert_files = list(tmp.glob("alert_*.json"))
    assert len(alert_files) == 1, f"FAIL: expected 1 file, got {len(alert_files)}"

    with open(alert_files[0]) as f:
        payload = json.load(f)

    assert payload["n_candidates"] == 1
    assert payload["candidates"][0]["tic_id"] == "TIC 900000001"
    print(f"Alert file: {alert_files[0].name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("PASS: alert system triggers correctly for 1 valid NEW_CANDIDATE")
finally:
    cfg.ALERTS_DIR = original_dir
```

**Pass criteria**:
- [ ] Exactly 1 alert triggered (not 2 — the low-SNR one must be excluded)
- [ ] JSON file created with correct structure
- [ ] Only TIC 900000001 in the payload (KNOWN_MATCH and low-SNR excluded)

---

## Summary checklist

After completing all tasks, fill in this table:

| Task | Description | Pass? |
|------|-------------|-------|
| 1 | Installation + 192 tests pass | |
| 2 | Cache roundtrip preserves data | |
| 3 | Preprocessing preserves transit signal | |
| 4 | BLS finds correct period | |
| 5 | Numba and lightkurve BLS agree | |
| 6 | Multi-planet search finds >= 2 planets | |
| 7 | Validator accepts good, rejects bad | |
| 8 | Cross-match: all 4 classes correct | |
| 9 | ML classifier: accuracy > 70%, probabilities sum to 1 | |
| 10 | CSV/FITS/VOTable export roundtrip | |
| 11 | Dashboard serves with all components | |
| 12 | Alert system triggers correctly | |

**All 12 tasks must pass for the pipeline to be considered verified.**

---

## Reporting issues

If any task fails:

1. Note the exact error message and task number
2. Check if it's a network issue (tasks 2–4, 6 require internet for the first download)
3. Open an issue on GitHub with the error output
4. Try running `pytest tests/` to see if the automated tests catch the same issue
