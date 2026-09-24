"""
tests/test_model_architectures.py
===================================
Unit tests for all model architectures.

Run with: pytest tests/
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip if TensorFlow not installed
try:
    from module1_refinement.model_architectures import (
        build_lstm, build_bilstm, build_gru, build_transformer, build_model
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow not installed"
)

SEQUENCE_LENGTH = 30
FEATURE_DIM     = 258
NUM_CLASSES     = 10  # small for testing
BATCH_SIZE      = 4


def make_dummy_batch():
    X = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    return X, y


class TestModelArchitectures:

    def test_lstm_output_shape(self):
        model = build_lstm(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH,
                           feature_dim=FEATURE_DIM)
        X, _ = make_dummy_batch()
        out = model.predict(X, verbose=0)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected ({BATCH_SIZE},{NUM_CLASSES}), got {out.shape}"

    def test_bilstm_output_shape(self):
        model = build_bilstm(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH,
                              feature_dim=FEATURE_DIM)
        X, _ = make_dummy_batch()
        out = model.predict(X, verbose=0)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_gru_output_shape(self):
        model = build_gru(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH,
                          feature_dim=FEATURE_DIM)
        X, _ = make_dummy_batch()
        out = model.predict(X, verbose=0)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_transformer_output_shape(self):
        model = build_transformer(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH,
                                   feature_dim=FEATURE_DIM)
        X, _ = make_dummy_batch()
        out = model.predict(X, verbose=0)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_softmax_probabilities_sum_to_one(self):
        model = build_lstm(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH,
                           feature_dim=FEATURE_DIM)
        X, _ = make_dummy_batch()
        out = model.predict(X, verbose=0)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(BATCH_SIZE), atol=1e-5)

    def test_factory_function(self):
        for arch in ["lstm", "bilstm", "gru", "transformer"]:
            model = build_model(arch, num_classes=NUM_CLASSES)
            assert model is not None

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError):
            build_model("invalid_arch")
