"""
tests/test_ensemble.py
======================
Several seed models saved as one averaging model (src/ensemble.py), the
threshold that goes with it, and retraining the seeds side by side.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.live_session import CONFIDENCE_THRESHOLD, LiveSession  # noqa: E402
from src import ensemble, retrain  # noqa: E402


def _tiny(seed):
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    inp = keras.Input((30, 258))
    x = keras.layers.GlobalAveragePooling1D()(inp)
    return keras.Model(inp, keras.layers.Dense(5, activation="softmax", name="output")(x))


def test_ensemble_file_averages_its_members(tmp_path):
    pytest.importorskip("tensorflow")
    from tensorflow import keras
    paths = []
    for seed in (1, 2, 3):
        _tiny(seed).save(str(tmp_path / f"m{seed}.h5"))
        paths.append(tmp_path / f"m{seed}.h5")
    ensemble.combine_files(paths, tmp_path / "ens.h5")

    ens = keras.models.load_model(str(tmp_path / "ens.h5"), compile=False)
    x = np.random.default_rng(0).random((4, 30, 258)).astype(np.float32)
    expected = np.mean([keras.models.load_model(str(p), compile=False).predict(x, verbose=0)
                        for p in paths], axis=0)
    assert np.allclose(ens.predict(x, verbose=0), expected, atol=1e-6)
    assert ens.input_shape == (None, 30, 258) and ens.output_shape == (None, 5)
    assert len(ensemble.members(ens)) == 3
    assert ensemble.default_threshold(ens) == ensemble.ENSEMBLE_THRESHOLD


def test_single_model_keeps_its_threshold():
    pytest.importorskip("tensorflow")
    model = _tiny(0)
    assert ensemble.members(model) == [model]
    assert ensemble.default_threshold(model) == ensemble.SINGLE_THRESHOLD == 0.65


class _Engine:
    threshold = 0.445

    def best_guess(self, sequence):
        return "selam", "ሰላም", 0.5


def test_live_session_uses_the_engines_threshold():
    assert LiveSession(_Engine()).threshold == 0.445
    assert LiveSession(_Engine(), threshold=0.8).threshold == 0.8
    assert LiveSession(None).threshold == CONFIDENCE_THRESHOLD   # an engine without one


def test_seeds_run_side_by_side_with_their_names(tmp_path):
    py = sys.executable
    cmds = {"seed 1": [py, "-c", "print('one')"], "seed 2": [py, "-c", "print('two'); print('done')"]}
    lines = list(retrain._stream_parallel(cmds, tmp_path, threads=1))
    assert sorted(lines) == ["[seed 1] one", "[seed 2] done", "[seed 2] two"]
    assert lines.index("[seed 2] two") < lines.index("[seed 2] done")


def test_a_failed_seed_fails_the_retrain(tmp_path):
    py = sys.executable
    cmds = {"seed 1": [py, "-c", "print('ok')"], "seed 2": [py, "-c", "import sys; sys.exit(3)"]}
    with pytest.raises(RuntimeError, match="seed 2"):
        list(retrain._stream_parallel(cmds, tmp_path, threads=1))


def test_each_seed_gets_its_own_config(tmp_path):
    import yaml
    cfg = yaml.safe_load(retrain._write_config(tmp_path / "seed_7", seed=7).read_text(encoding="utf-8"))
    assert cfg["training"]["seed"] == 7
    assert cfg["output"]["model_dir"] == str(tmp_path / "seed_7")
    assert cfg["finetune"]["compare_architectures"] is False


def test_mirror_matrix_is_the_mirror():
    from src.landmark_layout import mirror
    x = np.random.default_rng(1).normal(size=(30, 258)).astype(np.float32)
    assert np.allclose(x @ ensemble.mirror_matrix(258), mirror(x))


def test_with_mirror_averages_the_sign_and_its_mirror_image():
    pytest.importorskip("tensorflow")
    from src.landmark_layout import mirror
    model = _tiny(5)
    x = np.random.default_rng(2).normal(size=(3, 30, 258)).astype(np.float32)
    expected = (model.predict(x, verbose=0)
                + model.predict(np.stack([mirror(s) for s in x]), verbose=0)) / 2
    assert np.allclose(ensemble.with_mirror(model).predict(x, verbose=0), expected, atol=1e-6)


def test_only_an_ensemble_gets_the_mirrored_copy(tmp_path):
    pytest.importorskip("tensorflow")
    from tensorflow import keras
    single = _tiny(0)
    paths = []
    for seed in (1, 2):
        _tiny(seed).save(str(tmp_path / f"m{seed}.h5"))
        paths.append(tmp_path / f"m{seed}.h5")
    ensemble.combine_files(paths, tmp_path / "ens.h5")
    ens = keras.models.load_model(str(tmp_path / "ens.h5"), compile=False)
    assert ensemble.uses_mirror(ens) and not ensemble.uses_mirror(single)

