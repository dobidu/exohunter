# Machine Learning Classification Guide

This guide walks through ExoHunter's ML classification system — from
downloading training data to running predictions on new candidates.
It is designed for students learning to apply machine learning to a
real scientific problem.

---

## 1. The problem

After the BLS algorithm detects a periodic transit-like signal, we need
to decide: is this a **real planet**, an **eclipsing binary**, or some
other **false positive**?

The rule-based validator (6 criteria in `detection/validator.py`) catches
obvious cases, but borderline signals require a more nuanced approach.
A Random Forest classifier can learn subtle patterns from thousands of
labeled examples that hard-coded thresholds miss.

---

## 2. Training data: the Kepler KOI catalog

### Why Kepler?

NASA's Kepler mission observed the same patch of sky for 4 years
(2009–2013), producing the most thoroughly vetted catalog of transit
candidates in existence. Each candidate has a **disposition** — a label
assigned by experts after extensive follow-up:

| Disposition | Meaning | Count (~) |
|-------------|---------|-----------|
| `CONFIRMED` | Verified exoplanet | 2,783 |
| `FALSE POSITIVE` | Not a planet | 4,847 |
| `CANDIDATE` | Awaiting confirmation | 1,934 |

The CONFIRMED and FALSE POSITIVE entries give us ~7,600 labeled
examples — enough for a solid Random Forest. Kepler features (period,
depth, duration, SNR) map directly to TESS features, so a model trained
on Kepler generalizes to TESS candidates.

### Three-class labels

We split FALSE POSITIVE further using the `koi_fpflag_ss` flag (stellar
eclipse), which identifies eclipsing binaries:

| Class | Source | What it means |
|-------|--------|---------------|
| `planet` | CONFIRMED | Genuine exoplanet transit |
| `eclipsing_binary` | FALSE POSITIVE with `koi_fpflag_ss=1` | Two stars eclipsing each other |
| `false_positive` | FALSE POSITIVE with `koi_fpflag_ss=0` | Other false alarm (instrumental, blended, etc.) |

### Data sources

Both datasets are defined in `data/datasets_sources.json`:

| Dataset | Source | Download |
|---------|--------|----------|
| Kepler KOI cumulative | NASA Exoplanet Archive (TAP API) | ~9,500 candidates |
| ExoFOP-TESS TOI | ExoFOP / IPAC | ~7,900 TOIs (for TESS-specific validation) |

---

## 3. Step-by-step: training the classifier

### 3.1 Download the datasets

```bash
python scripts/download_training_data.py
```

This reads the URLs from `data/datasets_sources.json` and saves:
- `data/datasets/kepler_koi.csv` — Kepler KOI cumulative table
- `data/datasets/exofop_toi.csv` — ExoFOP-TESS TOI catalog

To re-download (e.g., to get updated catalogs):

```bash
python scripts/download_training_data.py --force
```

### 3.2 Understand the features

The classifier uses 10 features, mapped from the Kepler KOI columns to
ExoHunter's internal representation:

| Feature | Kepler column | Unit | What it measures |
|---------|---------------|------|-----------------|
| `period` | `koi_period` | days | Orbital period |
| `depth` | `koi_depth` | fractional | Transit depth (how much light is blocked) |
| `duration` | `koi_duration` | days | How long the transit lasts |
| `snr` | `koi_model_snr` | dimensionless | Signal-to-noise ratio |
| `impact_param` | `koi_impact` | dimensionless | How close to center the planet crosses the star (0 = center, 1 = edge) |
| `stellar_teff` | `koi_steff` | K | Star surface temperature |
| `stellar_logg` | `koi_slogg` | cgs | Star surface gravity (proxy for size) |
| `stellar_radius` | `koi_srad` | R_sun | Star radius in solar units |
| `duration_period_ratio` | derived | dimensionless | duration / period (physical constraint) |
| `depth_log` | derived | dimensionless | log10(depth) — compresses the range |

### 3.3 Train the model

```bash
python scripts/train_classifier.py
```

This will:
1. Load `data/datasets/kepler_koi.csv`
2. Map dispositions to 3-class labels
3. Drop CANDIDATE rows (no ground truth)
4. Run 5-fold stratified cross-validation
5. Print a classification report and confusion matrix
6. Fit the final model on all training data
7. Print feature importances
8. Save the model to `data/models/transit_classifier.joblib`

**Expected output** (cross-validation):

```
              precision    recall  f1-score   support

eclipsing_binary       0.XX      0.XX      0.XX       XXXX
false_positive         0.XX      0.XX      0.XX       XXXX
planet                 0.XX      0.XX      0.XX       XXXX

accuracy                                   0.XX       XXXX
```

### 3.4 Validate on TESS data (optional)

```bash
python scripts/train_classifier.py --validate-tess
```

This additionally:
1. Loads the ExoFOP-TESS TOI catalog
2. Runs the trained model on TESS candidates with known dispositions
3. Prints a binary comparison (planet vs. not-planet) since ExoFOP
   doesn't distinguish eclipsing binaries from other false positives

### 3.5 Custom hyperparameters

```bash
python scripts/train_classifier.py --n-estimators 500 --max-depth 30 --cv-folds 10
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--n-estimators` | 300 | More trees = better accuracy but slower |
| `--max-depth` | 20 | Deeper trees = more complex patterns but risk overfitting |
| `--cv-folds` | 5 | More folds = more robust estimate but slower |

---

## 4. Using the classifier in the pipeline

### 4.1 Classify batch results

```bash
python scripts/run_batch.py --sector 56 --classify --limit 20
```

The `--classify` flag:
1. Loads the trained model from `data/models/transit_classifier.joblib`
2. After BLS detection and validation, extracts features from each candidate
3. Runs `model.predict()` and `model.predict_proba()`
4. Adds `ml_class` and `ml_prob_planet` columns to the output CSV

If no trained model exists, the flag is silently skipped with a warning.

### 4.2 View results in the dashboard

```bash
python scripts/run_dashboard.py
```

The dashboard shows:
- **ML Class** column — predicted label (planet, eclipsing_binary, false_positive)
- **ML P(planet)** column — probability that the candidate is a real planet (0.0–1.0)
- Color-coded rows: green tint for `planet`, red tint for `eclipsing_binary`

### 4.3 Programmatic usage

```python
from exohunter.classification.model import load_model, classify_candidates
from exohunter.classification.features import candidate_to_features
import pandas as pd

# Load the trained model
model = load_model()

# Extract features from an ExoHunter candidate
features = candidate_to_features(candidate, validation)
features_df = pd.DataFrame([features])

# Classify
results = classify_candidates(model, features_df)
print(results["ml_class"].iloc[0])        # "planet", "eclipsing_binary", or "false_positive"
print(results["ml_prob_planet"].iloc[0])   # 0.0 – 1.0
```

---

## 5. How the model works

### 5.1 The pipeline

The classifier is a scikit-learn `Pipeline` with three stages:

```
Raw features (10 values, may contain NaN)
    │
    ▼
[SimpleImputer] ─── fills NaN with column medians
    │
    ▼
[StandardScaler] ── normalizes to mean=0, std=1
    │
    ▼
[RandomForestClassifier] ── 300 trees, max_depth=20, balanced class weights
    │
    ▼
Prediction: class label + per-class probabilities
```

### 5.2 Why Random Forest?

| Property | Why it matters for this problem |
|----------|-------------------------------|
| **Handles mixed features** | Our features span different scales (period in days, depth in fractional, temperature in Kelvin) |
| **Robust to outliers** | Transit candidates have noisy, heterogeneous measurements |
| **Interpretable** | Feature importances show which features matter most — students can understand *why* the model makes decisions |
| **No hyperparameter sensitivity** | Works well with defaults; doesn't need extensive tuning like neural networks |
| **Handles NaN** | Combined with SimpleImputer, gracefully handles missing stellar parameters |

### 5.3 Class balancing

The training data is imbalanced (~2,800 planets vs. ~4,800 false
positives). We use `class_weight="balanced"` in the Random Forest,
which automatically adjusts sample weights inversely proportional to
class frequency. This prevents the model from being biased toward the
majority class.

### 5.4 Feature importance

After training, the model reports which features are most important
for classification. Typical results:

1. **snr** — signal strength is the strongest predictor
2. **depth_log** — transit depth (log-scale) separates planets from binaries
3. **impact_param** — grazing transits (high impact) are often false positives
4. **duration_period_ratio** — physical constraint from orbital mechanics
5. **stellar_radius** — larger stars produce shallower transits for the same planet size

---

## 6. Feature mapping: Kepler → ExoHunter

When the classifier runs on ExoHunter candidates (TESS data), the
features are extracted differently than from the Kepler catalog:

| Feature | Kepler KOI source | ExoHunter source |
|---------|-------------------|-----------------|
| `period` | `koi_period` column | `TransitCandidate.period` |
| `depth` | `koi_depth / 1e6` | `TransitCandidate.depth` |
| `duration` | `koi_duration / 24` | `TransitCandidate.duration` |
| `snr` | `koi_model_snr` | `TransitCandidate.snr` |
| `impact_param` | `koi_impact` | Estimated from V-shape test (0.2 if box-like, 0.7 if V-shaped) |
| `stellar_teff` | `koi_steff` | TIC metadata or default 5778 K (solar) |
| `stellar_logg` | `koi_slogg` | TIC metadata or default 4.44 (solar) |
| `stellar_radius` | `koi_srad` | TIC metadata or default 1.0 R_sun |
| `duration_period_ratio` | derived | `duration / period` |
| `depth_log` | derived | `log10(depth)` |

The V-shape → impact parameter mapping is an approximation. A more
accurate estimate would require fitting a limb-darkened transit model,
which is a natural extension for students.

---

## 7. Architecture

```
data/
├── datasets_sources.json        # URLs for Kepler KOI + ExoFOP (committed)
├── datasets/
│   ├── kepler_koi.csv           # Downloaded training data (gitignored)
│   └── exofop_toi.csv           # Downloaded validation data (gitignored)
└── models/
    └── transit_classifier.joblib # Trained model (gitignored)

exohunter/classification/
├── __init__.py
├── datasets.py                  # Download + prepare training DataFrames
├── features.py                  # Feature extraction (candidate → vector)
└── model.py                     # Train, save, load, predict

scripts/
├── download_training_data.py    # Step 1: download datasets
└── train_classifier.py          # Step 2: train the model
```

---

## 8. Extending the classifier

### For students

- **Add features**: Try adding `n_transits`, `cdpp`, or phase curve
  shape features. Does accuracy improve?
- **Try other algorithms**: Replace RandomForest with GradientBoosting,
  SVM, or a simple neural network. Compare cross-validation scores.
- **Feature selection**: Use `sklearn.feature_selection` to find the
  minimal feature set that maintains accuracy.
- **Confusion matrix analysis**: Which class is hardest to classify?
  Which false positives look most like planets?

### Advanced

- **CNN on phase curves**: Instead of tabular features, use the
  phase-folded light curve directly as input to a 1D convolutional
  neural network (following Shallue & Vanderburg 2018).
- **Transfer learning**: Train on Kepler, fine-tune on TESS (following
  ExoMiner++ approach).
- **Active learning**: Use the model's uncertainty (`predict_proba`) to
  prioritize which CANDIDATE entries to label next.

---

## 9. References

- Kovacs, G., Zucker, S., & Mazeh, T. (2002). "A box-fitting algorithm
  in the search for periodic transits." *A&A*, 391, 369-377.
- Shallue, C. J. & Vanderburg, A. (2018). "Identifying Exoplanets with
  Deep Learning." *AJ*, 155, 94.
  [GitHub](https://github.com/google-research/exoplanet-ml)
- Thompson, S. E. et al. (2018). "Planetary Candidates Observed by
  Kepler. VIII." *ApJS*, 235, 38. (Kepler DR25 KOI catalog)
- Valizadegan, H. et al. (2022). "ExoMiner: A Highly Accurate and
  Explainable Deep Learning Classifier." *ApJ*, 926, 120.
- Yu, L. et al. (2019). "Identifying Exoplanets with Deep Learning.
  III. Automated Triage and Vetting of TESS Candidates." *AJ*, 158, 25.
  [GitHub](https://github.com/yuliang419/Astronet-Triage)
