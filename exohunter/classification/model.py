"""Random Forest classifier for transit candidates.

Provides functions to train, save, load, and run inference with a
scikit-learn RandomForestClassifier. The model predicts three classes:

    - ``planet`` — genuine exoplanet transit
    - ``eclipsing_binary`` — eclipsing binary star system
    - ``false_positive`` — other false positive (instrumental, blended, etc.)

The model is trained on the Kepler KOI cumulative table (~7,500 labeled
examples) and can be validated on the ExoFOP-TESS TOI catalog.

Usage::

    from exohunter.classification.model import load_model, classify_candidates

    model = load_model()
    results = classify_candidates(model, candidates_df)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exohunter import config
from exohunter.classification.features import FEATURE_COLUMNS
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# Default model path
MODEL_PATH: Path = config.MODELS_DIR / "transit_classifier.joblib"

# Class labels in consistent order
CLASS_LABELS: list[str] = ["planet", "eclipsing_binary", "false_positive"]


def build_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = 20,
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn pipeline: imputer → scaler → RandomForest.

    The pipeline handles missing values (NaN) via median imputation,
    standardizes features, and applies the classifier.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (``None`` for unlimited).
        random_state: Random seed for reproducibility.

    Returns:
        A scikit-learn ``Pipeline`` ready for ``.fit()`` and ``.predict()``.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def train(
    train_df: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = 20,
    cv_folds: int = 5,
) -> Pipeline:
    """Train the classifier on a labeled DataFrame.

    Performs stratified k-fold cross-validation to estimate
    performance, then fits the final model on all training data.

    Args:
        train_df: DataFrame with ``FEATURE_COLUMNS`` + ``label``.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        cv_folds: Number of cross-validation folds.

    Returns:
        The fitted ``Pipeline``.
    """
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values

    logger.info(
        "Training RandomForest (%d trees, max_depth=%s) on %d examples",
        n_estimators, max_depth, len(X),
    )

    # Cross-validation to estimate performance
    pipeline = build_pipeline(n_estimators, max_depth)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    logger.info("Running %d-fold stratified cross-validation...", cv_folds)
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)

    report = classification_report(y, y_pred_cv, zero_division=0)
    logger.info("Cross-validation results:\n%s", report)

    cm = confusion_matrix(y, y_pred_cv, labels=CLASS_LABELS)
    logger.info("Confusion matrix (rows=true, cols=predicted):\n%s", cm)

    # Fit final model on all data
    logger.info("Fitting final model on all %d examples...", len(X))
    pipeline.fit(X, y)

    return pipeline


def save_model(pipeline: Pipeline, path: Path | None = None) -> Path:
    """Save a trained model to disk using joblib.

    Args:
        pipeline: The fitted scikit-learn pipeline.
        path: Output path. Defaults to ``data/models/transit_classifier.joblib``.

    Returns:
        Path to the saved file.
    """
    import joblib

    if path is None:
        path = MODEL_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: Path | None = None) -> Pipeline:
    """Load a trained model from disk.

    Args:
        path: Path to the ``.joblib`` file. Defaults to
            ``data/models/transit_classifier.joblib``.

    Returns:
        The loaded scikit-learn ``Pipeline``.

    Raises:
        FileNotFoundError: If no trained model exists.
    """
    import joblib

    if path is None:
        path = MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. "
            f"Run: python scripts/train_classifier.py"
        )

    pipeline = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return pipeline


def classify_candidates(
    pipeline: Pipeline,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the classifier on a DataFrame of candidate features.

    Args:
        pipeline: A fitted scikit-learn pipeline (from ``load_model``).
        features_df: DataFrame with columns matching ``FEATURE_COLUMNS``.

    Returns:
        A DataFrame with columns: ``ml_class`` (predicted label) and
        ``ml_prob_planet``, ``ml_prob_eb``, ``ml_prob_fp``
        (per-class probabilities).
    """
    X = features_df[FEATURE_COLUMNS].values

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    # Map probability columns to class names
    classes = list(pipeline.classes_)
    prob_cols = {}
    for i, cls in enumerate(classes):
        short = {"planet": "planet", "eclipsing_binary": "eb", "false_positive": "fp"}
        col_name = f"ml_prob_{short.get(cls, cls)}"
        prob_cols[col_name] = probabilities[:, i]

    result = pd.DataFrame({
        "ml_class": predictions,
        **prob_cols,
    })

    # Log summary
    for cls in CLASS_LABELS:
        count = int(np.sum(predictions == cls))
        if count > 0:
            logger.info("  Classified %d as %s", count, cls)

    return result
