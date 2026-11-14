"""
src/retrain.py
==============
Retrain the model on everything in data/landmarks (used by the Retrain tab of
app/web_app.py), without touching the model the live app uses:

  1. preprocess.py and split_dataset.py rebuild the dataset, holding out the
     test signer (s04 once Menase's recordings are in, until then Gonfa g07)
  2. finetune.py trains the ISLR-pretrained LSTM into models/candidate/
  3. A team member compares its scores with the current model's and calls
     promote() to make it the live model; the old one is kept as
     models/finetuned/previous_model.h5

Authors: Estifanos Behailu
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
CANDIDATE_DIR = ROOT / "models" / "candidate"
LIVE_DIR = ROOT / "models" / "finetuned"
PRETRAINED = ROOT / "models" / "pretrained" / "islr_lstm.h5"
TEST_SIGNERS = ["s04", "g07"]   # first one present in the data is held out


def holdout_signer(signer_ids_path: Path = ROOT / "data" / "processed" / "signer_ids.npy") -> str:
    present = set(np.load(signer_ids_path, allow_pickle=True).tolist())
    return next((s for s in TEST_SIGNERS if s in present), TEST_SIGNERS[-1])


def _write_config(out_dir: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "module2_finetune" / "config.yaml").read_text(encoding="utf-8"))
    cfg["pretrained"]["path"] = str(PRETRAINED)
    cfg["pretrained"]["architecture"] = "lstm"
    cfg["finetune"]["compare_architectures"] = False
    cfg["output"]["model_dir"] = str(out_dir)
    path = out_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _stream(cmd, cwd: Path) -> Iterator[str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        yield line.rstrip("\n")
    if proc.wait() != 0:
        raise RuntimeError(f"{Path(cmd[1]).name} failed (exit code {proc.returncode})")


def run(out_dir: Path = CANDIDATE_DIR) -> Iterator[str]:
    """Rebuild the dataset and train a candidate model, yielding log lines."""
    py = sys.executable
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    yield from _stream([py, "scripts/preprocess.py"], ROOT)
    holdout = holdout_signer()
    yield f">>> Test signer (never trained on): {holdout}"
    yield from _stream([py, "scripts/split_dataset.py", "--holdout_signer", holdout], ROOT)
    started = time.time()
    yield from _stream([py, "finetune.py", "--config", str(_write_config(out_dir)),
                        "--arch", "lstm"], ROOT / "module2_finetune")
    vocab = json.loads((ROOT / "data" / "vocabulary.json").read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(json.dumps({
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "minutes": round((time.time() - started) / 60, 1),
        "test_signer": holdout,
        "words": vocab["metadata"]["total_signs"],
        "clips": int(len(np.load(ROOT / "data" / "processed" / "y.npy"))),
    }, indent=2) + "\n", encoding="utf-8")
    yield ">>> Done. Compare the scores, then press 'Use this model' to switch the live app to it."


def scores(model_dir: Path) -> Optional[Dict]:
    """The model's scores and training facts, or None if there is no model there."""
    if not (model_dir / "best_model.h5").exists():
        return None
    out = {}
    summary = model_dir / "summary.json"
    if summary.exists():
        out.update(json.loads(summary.read_text(encoding="utf-8")))
    results = model_dir / "all_results.json"
    if results.exists():
        r = json.loads(results.read_text(encoding="utf-8"))[0]
        out["unseen_signer"] = r.get("signer_independent", {}).get("accuracy")
        out["seen_signers"] = r.get("signer_dependent", {}).get("accuracy")
    return out


def promote(candidate: Path = CANDIDATE_DIR, live: Path = LIVE_DIR) -> None:
    """Make the candidate the live model, keeping the old one as previous_model.h5."""
    if not (candidate / "best_model.h5").exists():
        raise FileNotFoundError("There is no new model yet; press Retrain first.")
    live.mkdir(parents=True, exist_ok=True)
    if (live / "best_model.h5").exists():
        shutil.copy2(live / "best_model.h5", live / "previous_model.h5")
    for name in ("best_model.h5", "all_results.json", "summary.json", "comparison_results.csv"):
        if (candidate / name).exists():
            shutil.copy2(candidate / name, live / name)
