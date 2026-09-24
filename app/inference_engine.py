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

        # Warm up (first call is slower due to graph compilation)
        dummy = np.zeros((1, 30, 258), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        print("Model ready.\n")

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
        # Normalize
        seq = SequenceBuilder.normalize_sequence(sequence)
        inp = seq[np.newaxis, ...]  # (1, T, D)

        # Inference
        probs    = self.model.predict(inp, verbose=0)[0]  # (num_classes,)
        class_id = int(np.argmax(probs))
        conf     = float(probs[class_id])

        if conf < self.threshold:
            return None, None, conf

        label   = self.vocab.id_to_label(class_id)
        amharic = self.vocab.id_to_amharic(class_id)
        return label, amharic, conf

    def predict_top_k(self, sequence: np.ndarray, k: int = 3):
        """Return top-k predictions as list of (label, amharic, confidence)."""
        seq  = SequenceBuilder.normalize_sequence(sequence)
        inp  = seq[np.newaxis, ...]
        probs = self.model.predict(inp, verbose=0)[0]
        return self.vocab.top_k_prediction(probs, k=k)
