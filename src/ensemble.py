"""
src/ensemble.py
===============
Several models trained the same way but with different random seeds, saved
as ONE Keras model that averages their probabilities. The apps load it like
any other model, and the team still shares a single best_model.h5.

Why: leave-one-signer-out over the four team signers (2026-11-26) gave 73.4%
top-1 for one model (mean of 5 seeds) and 79.7% for the average of the same 5
models, better for every signer (s01 83.8 -> 89.5, s02 70.8 -> 77.7, s03 77.7
-> 83.5, s04 61.4 -> 68.0); top-3 88.6% -> 93.1%.

An average of several models is less extreme than one model, so it needs a
lower confidence threshold for the same share of "Not sure": 0.43 for 5
models gives 9.1% "Not sure", like 0.65 for one model (8.6%). That threshold
was chosen from the confidences alone, not from which guesses were right.
"""

from pathlib import Path
from typing import List, Sequence

SINGLE_THRESHOLD = 0.65
ENSEMBLE_THRESHOLD = 0.43


def combine(models: Sequence, name: str = "ensemble"):
    """One Keras model whose output is the average of these models' outputs."""
    from tensorflow import keras
    for i, m in enumerate(models):
        m.name = f"member_{i}"            # nested models need distinct names
    inp = keras.Input(shape=models[0].input_shape[1:], name="landmarks")
    out = keras.layers.Average(name="average")([m(inp) for m in models])
    return keras.Model(inp, out, name=name)


def combine_files(paths: Sequence[Path], out_path: Path) -> None:
    """Load the models in `paths` and save their average as out_path."""
    from tensorflow import keras
    models = [keras.models.load_model(str(p), compile=False) for p in paths]
    combine(models).save(str(out_path))


def members(model) -> List:
    """The models inside an ensemble made by combine(); [model] for a single model."""
    from tensorflow import keras
    inner = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    return inner if model.name == "ensemble" and inner else [model]


def default_threshold(model) -> float:
    """The confidence threshold that suits this model: lower for an ensemble."""
    return ENSEMBLE_THRESHOLD if len(members(model)) > 1 else SINGLE_THRESHOLD
