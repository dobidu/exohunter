"""Tests for the CNN transit classifier.

Tests phase curve generation, model architecture, training on
synthetic data, and prediction. All tests run on CPU — no GPU required.
PyTorch is required; tests are skipped if not installed.
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestPhaseCurveGeneration:
    """Test synthetic phase curve generation."""

    def test_output_shape(self) -> None:
        """Generated phase curve must have the correct number of bins."""
        from exohunter.classification.cnn import generate_synthetic_phase_curve, N_PHASE_BINS

        curve = generate_synthetic_phase_curve(
            period=5.0, depth=0.01, duration=0.1, seed=42,
        )
        assert curve.shape == (N_PHASE_BINS,)
        assert curve.dtype == np.float32

    def test_transit_dip_visible(self) -> None:
        """A deep transit must produce a visible dip near the center."""
        from exohunter.classification.cnn import generate_synthetic_phase_curve, N_PHASE_BINS

        curve = generate_synthetic_phase_curve(
            period=5.0, depth=0.02, duration=0.1, noise_std=0.0001, seed=42,
        )
        center = N_PHASE_BINS // 2
        wing = N_PHASE_BINS // 4

        center_flux = np.mean(curve[center - 3: center + 4])
        wing_flux = np.mean(curve[:wing])

        assert center_flux < wing_flux, "No transit dip at phase center"

    def test_batch_generation(self) -> None:
        """generate_training_phase_curves must produce correct shapes."""
        import pandas as pd
        from exohunter.classification.cnn import generate_training_phase_curves, N_PHASE_BINS

        df = pd.DataFrame([
            {"period": 5.0, "depth": 0.01, "duration": 0.1, "label": "planet"},
            {"period": 3.0, "depth": 0.02, "duration": 0.08, "label": "eclipsing_binary"},
            {"period": 8.0, "depth": 0.001, "duration": 0.05, "label": "false_positive"},
        ])

        X, y = generate_training_phase_curves(df, n_bins=N_PHASE_BINS)

        assert X.shape == (3, N_PHASE_BINS)
        assert y.shape == (3,)
        assert set(y) == {0, 1, 2}


class TestCNNArchitecture:
    """Test the CNN model structure."""

    def test_build_creates_module(self) -> None:
        """build_cnn must return a torch.nn.Module."""
        from exohunter.classification.cnn import build_cnn

        model = build_cnn()
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Forward pass must produce correct output shape."""
        from exohunter.classification.cnn import build_cnn, N_PHASE_BINS, CLASS_LABELS

        model = build_cnn()
        x = torch.randn(4, 1, N_PHASE_BINS)
        output = model(x)

        assert output.shape == (4, len(CLASS_LABELS))

    def test_output_not_all_same(self) -> None:
        """Different inputs must produce different outputs."""
        from exohunter.classification.cnn import build_cnn, N_PHASE_BINS

        model = build_cnn()
        x1 = torch.randn(1, 1, N_PHASE_BINS)
        x2 = torch.randn(1, 1, N_PHASE_BINS)

        out1 = model(x1).detach().numpy()
        out2 = model(x2).detach().numpy()

        assert not np.allclose(out1, out2)


class TestCNNTraining:
    """Test CNN training on tiny synthetic data."""

    @pytest.fixture(scope="class")
    def tiny_data(self):
        """Generate minimal training data."""
        import pandas as pd
        from exohunter.classification.cnn import generate_training_phase_curves

        rows = []
        for i in range(30):
            rows.append({"period": 5.0, "depth": 0.01 + i * 0.001, "duration": 0.1, "label": "planet"})
        for i in range(30):
            rows.append({"period": 3.0, "depth": 0.03 + i * 0.002, "duration": 0.15, "label": "eclipsing_binary"})
        for i in range(30):
            rows.append({"period": 8.0, "depth": 0.0005 + i * 0.0001, "duration": 0.05, "label": "false_positive"})

        return generate_training_phase_curves(pd.DataFrame(rows))

    def test_train_runs_without_error(self, tiny_data) -> None:
        """Training must complete without crashing."""
        from exohunter.classification.cnn import train_cnn

        X, y = tiny_data
        model = train_cnn(X, y, n_epochs=3, batch_size=16)

        assert model is not None
        assert hasattr(model, "forward")

    def test_predictions_are_valid(self, tiny_data) -> None:
        """Trained model must produce valid class predictions."""
        from exohunter.classification.cnn import train_cnn, classify_phase_curves, CLASS_LABELS

        X, y = tiny_data
        model = train_cnn(X, y, n_epochs=3, batch_size=16)

        results = classify_phase_curves(model, X[:5])

        assert len(results) == 5
        assert "ml_class" in results.columns
        for cls in results["ml_class"]:
            assert cls in CLASS_LABELS

    def test_probabilities_sum_to_one(self, tiny_data) -> None:
        """Per-class probabilities must sum to 1.0."""
        from exohunter.classification.cnn import train_cnn, classify_phase_curves

        X, y = tiny_data
        model = train_cnn(X, y, n_epochs=3, batch_size=16)

        results = classify_phase_curves(model, X[:5])
        prob_cols = ["ml_prob_planet", "ml_prob_eb", "ml_prob_fp"]
        row_sums = results[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestCNNPersistence:
    """Test model save/load."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Saved model must load and produce identical outputs."""
        from exohunter.classification.cnn import (
            build_cnn, save_cnn_model, load_cnn_model, N_PHASE_BINS,
        )

        # Build on CPU to avoid device mismatch
        model = build_cnn().cpu()
        model.eval()
        path = tmp_path / "test_cnn.pt"
        save_cnn_model(model, path)

        loaded = load_cnn_model(path)  # loads to CPU

        x = torch.randn(2, 1, N_PHASE_BINS)
        with torch.no_grad():
            out_orig = model(x).numpy()
            out_loaded = loaded(x).numpy()

        np.testing.assert_array_almost_equal(out_orig, out_loaded)

    def test_load_missing_raises(self, tmp_path) -> None:
        """Loading nonexistent model must raise FileNotFoundError."""
        from exohunter.classification.cnn import load_cnn_model

        with pytest.raises(FileNotFoundError):
            load_cnn_model(tmp_path / "missing.pt")


class TestCandidateToPhase:
    """Test conversion of light curve to CNN input."""

    def test_output_shape(self) -> None:
        """candidate_to_phase_curve must return correct shape."""
        from exohunter.classification.cnn import candidate_to_phase_curve, N_PHASE_BINS

        time = np.linspace(0, 90, 10000)
        flux = np.ones_like(time)

        curve = candidate_to_phase_curve(time, flux, period=5.0, epoch=2.5)

        assert curve.shape == (N_PHASE_BINS,)
        assert curve.dtype == np.float32
