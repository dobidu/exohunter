"""Tests for the ML classification module.

Tests feature extraction, model training (on synthetic data),
prediction, and the datasets module. All tests run offline —
no downloads from NASA archives.
"""

import numpy as np
import pandas as pd
import pytest

from exohunter.classification.features import (
    FEATURE_COLUMNS,
    candidate_to_features,
    candidates_to_dataframe,
)
from exohunter.classification.model import (
    CLASS_LABELS,
    build_pipeline,
    classify_candidates,
    train,
)
from tests.conftest import make_candidate, make_validation


# ---------------------------------------------------------------------------
# Synthetic training data for offline testing
# ---------------------------------------------------------------------------

def _make_synthetic_training_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic labeled data mimicking the Kepler KOI schema.

    Creates three clusters in feature space:
        - planet: moderate depth (~0.001), moderate SNR, low impact
        - eclipsing_binary: deep depth (~0.02), high SNR, high impact
        - false_positive: shallow depth (~0.0001), low SNR
    """
    rng = np.random.default_rng(seed)
    rows = []

    for label, depth_mean, snr_mean, impact_mean, count in [
        ("planet", 0.001, 20.0, 0.3, n),
        ("eclipsing_binary", 0.02, 40.0, 0.7, n),
        ("false_positive", 0.0001, 5.0, 0.1, n),
    ]:
        for _ in range(count):
            depth = max(1e-8, rng.normal(depth_mean, depth_mean * 0.3))
            rows.append({
                "period": rng.uniform(0.5, 20.0),
                "depth": depth,
                "duration": rng.uniform(0.02, 0.2),
                "snr": max(0.1, rng.normal(snr_mean, snr_mean * 0.3)),
                "impact_param": max(0.0, min(1.0, rng.normal(impact_mean, 0.2))),
                "stellar_teff": rng.normal(5500, 500),
                "stellar_logg": rng.normal(4.3, 0.3),
                "stellar_radius": max(0.3, rng.normal(1.0, 0.3)),
                "duration_period_ratio": 0.0,  # filled below
                "depth_log": 0.0,  # filled below
                "label": label,
            })

    df = pd.DataFrame(rows)
    df["duration_period_ratio"] = df["duration"] / df["period"]
    df["depth_log"] = np.log10(df["depth"].clip(lower=1e-10))
    return df


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Test conversion of ExoHunter candidates to ML feature vectors."""

    def test_all_feature_columns_present(self) -> None:
        """Feature dict must contain all expected columns."""
        c = make_candidate(period=10.0, depth=0.01, duration=0.2, snr=15.0)
        v = make_validation()

        features = candidate_to_features(c, v)

        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_feature_values_correct(self) -> None:
        """Feature values must match candidate properties."""
        c = make_candidate(period=5.0, depth=0.005, duration=0.1, snr=12.0)
        v = make_validation(v_shape_pass=True)

        features = candidate_to_features(c, v)

        assert features["period"] == 5.0
        assert features["depth"] == 0.005
        assert features["duration"] == 0.1
        assert features["snr"] == 12.0
        assert abs(features["duration_period_ratio"] - 0.02) < 1e-10
        assert abs(features["depth_log"] - np.log10(0.005)) < 1e-10

    def test_impact_param_from_vshape(self) -> None:
        """Impact parameter must reflect V-shape test result."""
        c = make_candidate()

        v_box = make_validation(v_shape_pass=True)
        v_vee = make_validation(v_shape_pass=False)

        feat_box = candidate_to_features(c, v_box)
        feat_vee = candidate_to_features(c, v_vee)

        assert feat_box["impact_param"] < feat_vee["impact_param"]

    def test_default_stellar_params(self) -> None:
        """Missing stellar params must default to solar values."""
        c = make_candidate()
        features = candidate_to_features(c)

        assert features["stellar_teff"] == 5778.0  # solar Teff
        assert features["stellar_logg"] == 4.44     # solar logg
        assert features["stellar_radius"] == 1.0    # solar radius

    def test_candidates_to_dataframe(self) -> None:
        """Batch conversion must produce correct DataFrame shape."""
        candidates = [
            (make_candidate(tic_id="A"), make_validation()),
            (make_candidate(tic_id="B"), make_validation()),
            (make_candidate(tic_id="C"), None),
        ]

        df = candidates_to_dataframe(candidates)

        assert len(df) == 3
        assert list(df.columns) == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Model tests (using synthetic data — no real datasets needed)
# ---------------------------------------------------------------------------

class TestModelTraining:
    """Test model training and prediction on synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_data(self) -> pd.DataFrame:
        return _make_synthetic_training_data(n=200)

    @pytest.fixture(scope="class")
    def trained_pipeline(self, synthetic_data):
        return train(synthetic_data, n_estimators=50, max_depth=10, cv_folds=3)

    def test_pipeline_structure(self) -> None:
        """Pipeline must have imputer, scaler, and classifier steps."""
        pipeline = build_pipeline()
        assert "imputer" in pipeline.named_steps
        assert "scaler" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_train_returns_fitted_pipeline(self, trained_pipeline) -> None:
        """train() must return a fitted pipeline that can predict."""
        assert hasattr(trained_pipeline, "predict")
        assert hasattr(trained_pipeline, "predict_proba")

    def test_predictions_are_valid_classes(self, trained_pipeline, synthetic_data) -> None:
        """Predictions must be one of the three class labels."""
        results = classify_candidates(trained_pipeline, synthetic_data)

        assert "ml_class" in results.columns
        for cls in results["ml_class"]:
            assert cls in CLASS_LABELS, f"Invalid class: {cls}"

    def test_probabilities_sum_to_one(self, trained_pipeline, synthetic_data) -> None:
        """Per-class probabilities must sum to approximately 1.0."""
        results = classify_candidates(trained_pipeline, synthetic_data)

        prob_cols = [c for c in results.columns if c.startswith("ml_prob_")]
        assert len(prob_cols) == 3

        row_sums = results[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_accuracy_above_chance(self, trained_pipeline, synthetic_data) -> None:
        """The model must beat random guessing (>33% for 3 classes)."""
        results = classify_candidates(trained_pipeline, synthetic_data)
        y_true = synthetic_data["label"].values
        y_pred = results["ml_class"].values

        accuracy = np.mean(y_true == y_pred)
        assert accuracy > 0.5, f"Accuracy {accuracy:.2%} is too low"

    def test_handles_nan_features(self, trained_pipeline) -> None:
        """Pipeline must handle NaN values via imputation (not crash)."""
        df = pd.DataFrame([{
            "period": 5.0, "depth": 0.001, "duration": 0.1,
            "snr": np.nan, "impact_param": np.nan,
            "stellar_teff": np.nan, "stellar_logg": np.nan,
            "stellar_radius": np.nan,
            "duration_period_ratio": 0.02, "depth_log": -3.0,
        }])

        results = classify_candidates(trained_pipeline, df)
        assert len(results) == 1
        assert results.iloc[0]["ml_class"] in CLASS_LABELS


class TestModelPersistence:
    """Test model save/load roundtrip."""

    def test_save_and_load(self, tmp_path) -> None:
        """A saved model must load and produce identical predictions."""
        from exohunter.classification.model import save_model, load_model

        data = _make_synthetic_training_data(n=100)
        pipeline = train(data, n_estimators=20, max_depth=5, cv_folds=2)

        model_path = tmp_path / "test_model.joblib"
        save_model(pipeline, model_path)

        loaded = load_model(model_path)

        # Predictions must match
        X = data[FEATURE_COLUMNS].values
        np.testing.assert_array_equal(
            pipeline.predict(X),
            loaded.predict(X),
        )

    def test_load_missing_raises(self, tmp_path) -> None:
        """Loading a nonexistent model must raise FileNotFoundError."""
        from exohunter.classification.model import load_model

        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")


class TestDatasetsSources:
    """Test the dataset sources configuration."""

    def test_sources_json_exists(self) -> None:
        """datasets_sources.json must exist in the project."""
        from exohunter.classification.datasets import load_sources_config

        sources = load_sources_config()
        assert "kepler_koi" in sources
        assert "exofop_toi" in sources

    def test_sources_have_required_fields(self) -> None:
        """Each source must have url, output, and description."""
        from exohunter.classification.datasets import load_sources_config

        sources = load_sources_config()
        for name, entry in sources.items():
            assert "url" in entry, f"{name} missing 'url'"
            assert "output" in entry, f"{name} missing 'output'"
            assert "description" in entry, f"{name} missing 'description'"
