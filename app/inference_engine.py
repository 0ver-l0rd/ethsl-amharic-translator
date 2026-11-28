"""
app/inference_engine.py
=======================
Loads the trained sign model and turns one landmark sequence (30 frames x 258
features) into a guess: the sign's label, its Amharic word and the model's
confidence. A guess below the confidence threshold is returned as None. The
threshold suits the model unless one is given: 0.65 for one model, 0.445 for
an ensemble of several (src/ensemble.py), whose averaged probabilities are
less extreme. An ensemble also classifies each sign mirrored and averages the
two answers (src/ensemble.with_mirror), which raised accuracy on new signers.
Used by both the web app and the desktop app.

The model runs through fast_predictor, a compiled function that is faster
than model.predict for one sequence at a time, and is warmed up when loaded
so the first real guess isn't slow.
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
from src.ensemble import default_threshold, members, uses_mirror, with_mirror
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
    confidence_threshold : float or None
        Minimum probability to report a prediction (None: the model's default,
        see src/ensemble.py).
    """

    def __init__(self, model_path: str, vocab, confidence_threshold: Optional[float] = None):
        self.vocab      = vocab

        print(f"Loading model: {model_path}")
        self.model = keras.models.load_model(model_path)
        self.model.summary()
        self.members = len(members(self.model))
        self.mirror = uses_mirror(self.model)
        self._infer = fast_predictor(with_mirror(self.model) if self.mirror else self.model)
        self.threshold = (default_threshold(self.model) if confidence_threshold is None
                          else confidence_threshold)
        if self.members > 1:
            print(f"Ensemble of {self.members} models, each also on the mirrored sign; "
                  f"confidence threshold {self.threshold:.3f}")

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
        label, amharic, conf = self.best_guess(sequence)
        if conf < self.threshold:
            return None, None, conf
        return label, amharic, conf

    def best_guess(self, sequence: np.ndarray) -> Tuple[str, str, float]:
        """(label, amharic, confidence) of the most likely sign, ignoring the threshold."""
        probs    = self._probs(sequence)
        class_id = int(np.argmax(probs))
        return self.vocab.id_to_label(class_id), self.vocab.id_to_amharic(class_id), float(probs[class_id])

    def predict_top_k(self, sequence: np.ndarray, k: int = 3):
        """Return top-k predictions as list of (label, amharic, confidence)."""
        return self.vocab.top_k_prediction(self._probs(sequence), k=k)
