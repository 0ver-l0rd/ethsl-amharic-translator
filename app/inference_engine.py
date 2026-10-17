"""
app/inference_engine.py
========================
Optimized inference pipeline wrapping the trained Keras model.

Authors: Emanuel Solomon
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from tensorflow import keras
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sequence_builder import SequenceBuilder
# Importing model_architectures registers the Transformer's custom layers,
# which load_model needs to deserialize a saved Transformer.
from module1_refinement.model_architectures import fast_predictor


class InferenceEngine:
    """
    Wraps a trained Keras model for real-time sign prediction.

    Parameters
    ----------
    model_path : str
        Path to the saved .h5 model.
    vocab : Vocabulary
        Vocabulary object for label lookup.
    confidence_threshold : float
        Minimum softmax probability to report a prediction.
    """

    def __init__(self, model_path: str, vocab, confidence_threshold: float = 0.65):
        self.vocab      = vocab
        self.threshold  = confidence_threshold

        print(f"Loading model: {model_path}")
        self.model = keras.models.load_model(model_path)
        self.model.summary()
        self._infer = fast_predictor(self.model)

        # Warm up (first call is slower due to graph compilation)
        _, seq_len, feat_dim = self.model.input_shape
        self._infer(np.zeros((1, seq_len, feat_dim), dtype=np.float32))
        print("Model ready.\n")

    def _probs(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize one (T, D) sequence and return its class probabilities."""
        seq = SequenceBuilder.normalize_sequence(sequence)
        return self._infer(seq[np.newaxis, ...])[0]  # (num_classes,)

    def predict(
        self,
        sequence: np.ndarray,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Run inference on a single landmark sequence.

        Parameters
        ----------
        sequence : np.ndarray of shape (T, D)

        Returns
        -------
        label : str or None
        amharic : str or None
        confidence : float
        """
        probs    = self._probs(sequence)
        class_id = int(np.argmax(probs))
        conf     = float(probs[class_id])

        if conf < self.threshold:
            return None, None, conf

        label   = self.vocab.id_to_label(class_id)
        amharic = self.vocab.id_to_amharic(class_id)
        return label, amharic, conf

    def predict_top_k(self, sequence: np.ndarray, k: int = 3):
        """Return top-k predictions as list of (label, amharic, confidence)."""
        return self.vocab.top_k_prediction(self._probs(sequence), k=k)
