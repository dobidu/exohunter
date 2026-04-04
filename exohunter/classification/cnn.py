"""1D Convolutional Neural Network classifier for transit phase curves.

Uses a simple 1D CNN architecture (following Shallue & Vanderburg 2018)
to classify phase-folded light curves into three classes:

    - ``planet`` — genuine exoplanet transit
    - ``eclipsing_binary`` — eclipsing binary star system
    - ``false_positive`` — other false positive

The CNN takes a 1D array of binned phase-folded flux values as input
(201 bins covering phase [-0.5, 0.5]) and outputs class probabilities.

Architecture::

    Input (1 x 201)
      -> Conv1d(1, 16, kernel=7, padding=3) -> ReLU -> MaxPool(2)
      -> Conv1d(16, 32, kernel=5, padding=2) -> ReLU -> MaxPool(2)
      -> Conv1d(32, 64, kernel=3, padding=1) -> ReLU -> AdaptiveAvgPool(1)
      -> Flatten -> Linear(64, 32) -> ReLU -> Dropout(0.3)
      -> Linear(32, 3)

Requires PyTorch::

    pip install -e ".[cnn]"

Usage::

    from exohunter.classification.cnn import load_cnn_model, classify_phase_curves

    model = load_cnn_model()
    results = classify_phase_curves(model, phase_curves_array)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

# Phase curve parameters
N_PHASE_BINS: int = 201
"""Number of bins in the phase-folded input (covers -0.5 to +0.5)."""

CLASS_LABELS: list[str] = ["planet", "eclipsing_binary", "false_positive"]

CNN_MODEL_PATH: Path = config.MODELS_DIR / "transit_cnn.pt"


def _check_torch() -> None:
    """Raise ImportError with a helpful message if PyTorch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for CNN classification. "
            "Install with: pip install -e '.[cnn]'"
        )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def _build_features_block():
    """Build the convolutional feature extraction layers."""
    _check_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(16, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


def _build_classifier_block():
    """Build the fully-connected classification head."""
    _check_torch()
    import torch.nn as nn

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, len(CLASS_LABELS)),
    )


def build_cnn():
    """Build the 1D CNN model.

    Returns:
        A ``torch.nn.Module`` instance (not yet trained).
    """
    _check_torch()
    import torch.nn as nn

    class TransitCNN(nn.Module):
        """1D CNN for transit phase curve classification."""

        def __init__(self):
            super().__init__()
            self.features = _build_features_block()
            self.classifier = _build_classifier_block()

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return TransitCNN()


# ---------------------------------------------------------------------------
# Phase curve generation
# ---------------------------------------------------------------------------

def generate_synthetic_phase_curve(
    period: float,
    depth: float,
    duration: float,
    noise_std: float = 0.001,
    n_bins: int = N_PHASE_BINS,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic binned phase-folded transit curve.

    Creates a trapezoidal transit model at phase=0, adds Gaussian
    noise, and bins into ``n_bins`` uniform phase bins.

    Args:
        period: Orbital period (used for duration/period ratio).
        depth: Fractional transit depth.
        duration: Transit duration in days.
        noise_std: Gaussian noise standard deviation.
        n_bins: Number of output phase bins.
        seed: Random seed for reproducibility.

    Returns:
        A 1D numpy array of length ``n_bins``.
    """
    from exohunter.detection.model import transit_model

    rng = np.random.default_rng(seed)

    # Generate a dense phase grid, then bin
    n_points = 5000
    phase_time = np.linspace(-0.5 * period, 0.5 * period, n_points)

    model_flux = transit_model(
        phase_time, period=period, epoch=0.0,
        duration=duration, depth=depth,
    )
    model_flux += rng.normal(0, noise_std, n_points)

    # Bin into n_bins uniform phase bins
    phase = phase_time / period  # [-0.5, 0.5]
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    binned = np.ones(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) > 0:
            binned[i] = np.mean(model_flux[mask])

    return binned.astype(np.float32)


def generate_training_phase_curves(
    catalog_df: pd.DataFrame,
    noise_range: tuple[float, float] = (0.0003, 0.003),
    n_bins: int = N_PHASE_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic phase curves for all entries in a training catalog.

    For each row in the catalog (from ``prepare_kepler_koi``), generates
    a synthetic phase curve using the catalog parameters.

    Args:
        catalog_df: DataFrame with ``period``, ``depth``, ``duration``,
            and ``label`` columns (from ``datasets.prepare_kepler_koi``).
        noise_range: Range of noise std to sample from (adds diversity).
        n_bins: Number of phase bins per curve.

    Returns:
        Tuple of ``(X, y)`` where ``X`` is shape ``(n_samples, n_bins)``
        and ``y`` is an integer label array (0=planet, 1=EB, 2=FP).
    """
    rng = np.random.default_rng(42)
    label_map = {name: i for i, name in enumerate(CLASS_LABELS)}

    X_list = []
    y_list = []

    for idx, row in catalog_df.iterrows():
        period = row.get("period", 5.0)
        depth = row.get("depth", 0.001)
        duration = row.get("duration", 0.05)
        label = row.get("label", "false_positive")

        if period <= 0 or depth <= 0 or duration <= 0:
            continue

        noise = rng.uniform(noise_range[0], noise_range[1])

        curve = generate_synthetic_phase_curve(
            period=period, depth=depth, duration=duration,
            noise_std=noise, n_bins=n_bins, seed=int(idx) + 1000,
        )
        X_list.append(curve)
        y_list.append(label_map.get(label, 2))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_split: float = 0.15,
) -> object:
    """Train the CNN on phase curve data.

    Args:
        X_train: Phase curves, shape ``(n_samples, n_bins)``.
        y_train: Integer labels, shape ``(n_samples,)``.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        validation_split: Fraction held out for validation.

    Returns:
        The trained PyTorch model.
    """
    _check_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training CNN on %s (%d samples, %d epochs)",
                device, len(X_train), n_epochs)

    # Train/val split
    n_val = int(len(X_train) * validation_split)
    indices = np.random.default_rng(42).permutation(len(X_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_t = torch.tensor(X_train[train_idx]).unsqueeze(1)  # (N, 1, n_bins)
    y_t = torch.tensor(y_train[train_idx])
    X_v = torch.tensor(X_train[val_idx]).unsqueeze(1)
    y_v = torch.tensor(y_train[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
    )

    model = build_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=len(CLASS_LABELS)).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights /= class_weights.sum()
    weights_tensor = torch.tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            val_out = model(X_v.to(device))
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            val_acc = np.mean(val_pred == y_v.numpy())

        avg_loss = total_loss / len(train_idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  Epoch %d/%d — loss: %.4f, val_acc: %.3f",
                epoch + 1, n_epochs, avg_loss, val_acc,
            )

    logger.info("Training complete. Final val_acc: %.3f", val_acc)
    return model


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_cnn_model(model: object, path: Path | None = None) -> Path:
    """Save a trained CNN model to disk.

    Args:
        model: The trained PyTorch model.
        path: Output path. Defaults to ``data/models/transit_cnn.pt``.

    Returns:
        Path to the saved file.
    """
    _check_torch()
    import torch

    if path is None:
        path = CNN_MODEL_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("CNN model saved to %s", path)
    return path


def load_cnn_model(path: Path | None = None) -> object:
    """Load a trained CNN model from disk.

    Args:
        path: Path to the ``.pt`` file.

    Returns:
        The loaded PyTorch model in eval mode.

    Raises:
        FileNotFoundError: If no trained model exists.
    """
    _check_torch()
    import torch

    if path is None:
        path = CNN_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No trained CNN model found at {path}. "
            f"Run: python scripts/train_cnn.py"
        )

    model = build_cnn()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info("CNN model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_phase_curves(
    model: object,
    phase_curves: np.ndarray,
) -> pd.DataFrame:
    """Run the CNN classifier on phase curve data.

    Args:
        model: A loaded CNN model (from ``load_cnn_model``).
        phase_curves: Array of shape ``(n_samples, n_bins)`` or
            ``(n_bins,)`` for a single sample.

    Returns:
        A DataFrame with ``ml_class``, ``ml_prob_planet``,
        ``ml_prob_eb``, ``ml_prob_fp`` columns.
    """
    _check_torch()
    import torch

    if phase_curves.ndim == 1:
        phase_curves = phase_curves[np.newaxis, :]

    X = torch.tensor(phase_curves, dtype=torch.float32).unsqueeze(1)  # (N, 1, bins)

    # Move input to the same device as the model
    device = next(model.parameters()).device
    X = X.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = [CLASS_LABELS[i] for i in probs.argmax(axis=1)]

    result = pd.DataFrame({
        "ml_class": predictions,
        "ml_prob_planet": probs[:, 0],
        "ml_prob_eb": probs[:, 1],
        "ml_prob_fp": probs[:, 2],
    })

    for cls in CLASS_LABELS:
        count = int(sum(1 for p in predictions if p == cls))
        if count > 0:
            logger.info("  CNN classified %d as %s", count, cls)

    return result


def candidate_to_phase_curve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    n_bins: int = N_PHASE_BINS,
) -> np.ndarray:
    """Convert a light curve to a binned phase curve for CNN input.

    Phase-folds the data at the given period/epoch and bins into
    ``n_bins`` uniform phase bins.

    Args:
        time: Array of timestamps.
        flux: Array of normalized flux values.
        period: Orbital period in days.
        epoch: Mid-transit time.
        n_bins: Number of output phase bins.

    Returns:
        A 1D numpy array of length ``n_bins``.
    """
    from exohunter.detection.model import phase_fold

    phase, flux_folded = phase_fold(time, flux, period, epoch)
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    binned = np.ones(n_bins, dtype=np.float32)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) >= 1:
            binned[i] = np.mean(flux_folded[mask])

    return binned
