"""
src/retrain.py
==============
Retrain the model on everything in data/landmarks (used by the Retrain tab of
app/web_app.py), without touching the model the live app uses:

  1. preprocess.py and split_dataset.py rebuild the dataset, holding out the
     test signer (s04, Menase; Gonfa g07 if there are no s04 clips)
  2. finetune.py trains the ISLR-pretrained LSTM five times, with different
     random seeds, side by side (models/candidate/seed_<n>/), and
     scripts/combine_models.py saves their average as one model,
     models/candidate/best_model.h5 (src/ensemble.py: 79.7% instead of 73.4%
     on unseen signers), with its scores
  3. A team member compares its scores with the current model's and calls
     promote() to make it the live model; the old one is kept as
     models/finetuned/previous_model.h5
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
CANDIDATE_DIR = ROOT / "models" / "candidate"
LIVE_DIR = ROOT / "models" / "finetuned"
PRETRAINED = ROOT / "models" / "pretrained" / "islr_lstm.h5"
TEST_SIGNERS = ["s04", "g07"]   # first one present in the data is held out
SEEDS = [42, 7, 1, 3, 11]       # one model per seed, averaged (src/ensemble.py)
# Where a model sent over Telegram ends up: Downloads (or a folder in it) or the repo folder
MODEL_SEARCH = [Path.home() / "Downloads", ROOT]


def find_live_model(live_dir: Path = LIVE_DIR, search: Iterable[Path] = MODEL_SEARCH
                    ) -> Tuple[Optional[Path], Optional[Path]]:
    """
    The live model, live_dir/best_model.h5, as (path, copied_from). If it is
    missing, the newest best_model*.h5 found in `search` (or a folder just
    inside one) is copied there first. (None, None) if there is none.
    """
    live = live_dir / "best_model.h5"
    if live.exists():
        return live, None
    found = [p for folder in search for pattern in ("best_model*.h5", "*/best_model*.h5")
             for p in Path(folder).glob(pattern) if p.is_file()]
    if not found:
        return None, None
    newest = max(found, key=lambda p: p.stat().st_mtime)
    live_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, live)
    return live, newest


def require_live_model() -> Path:
    """find_live_model() for the apps: says what it did, or how to get a model, and stops."""
    model, copied_from = find_live_model()
    if model is None:
        sys.exit("No model found. Get best_model.h5 from the team's Telegram group and put it in\n"
                 f"  {LIVE_DIR}\n(or just leave it in Downloads), then run this again.")
    if copied_from:
        print(f"Copied the model from {copied_from}\n  to {model}")
    return model


def holdout_signer(signer_ids_path: Path = ROOT / "data" / "processed" / "signer_ids.npy") -> str:
    present = set(np.load(signer_ids_path, allow_pickle=True).tolist())
    return next((s for s in TEST_SIGNERS if s in present), TEST_SIGNERS[-1])


def _write_config(out_dir: Path, seed: Optional[int] = None) -> Path:
    cfg = yaml.safe_load((ROOT / "module2_finetune" / "config.yaml").read_text(encoding="utf-8"))
    if seed is not None:
        cfg["training"]["seed"] = seed
    cfg["pretrained"]["path"] = str(PRETRAINED)
    cfg["pretrained"]["architecture"] = "lstm"
    cfg["finetune"]["compare_architectures"] = False
    cfg["output"]["model_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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


def _stream_parallel(cmds: Dict[str, list], cwd: Path, threads: int) -> Iterator[str]:
    """Run the commands side by side (each with `threads` CPU threads), yielding
    their output lines prefixed with their name."""
    env = {**os.environ, "TF_NUM_INTRAOP_THREADS": str(threads), "TF_NUM_INTEROP_THREADS": "2"}
    lines: "queue.Queue" = queue.Queue()
    procs = {}

    def read(name, proc):
        for line in proc.stdout:
            lines.put(f"[{name}] {line.rstrip()}")
        lines.put(None)

    for name, cmd in cmds.items():
        procs[name] = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=env)
        threading.Thread(target=read, args=(name, procs[name]), daemon=True).start()
    running = len(procs)
    while running:
        line = lines.get()
        if line is None:
            running -= 1
        else:
            yield line
    failed = [name for name, proc in procs.items() if proc.wait() != 0]
    if failed:
        raise RuntimeError(f"finetune.py failed for {', '.join(failed)}")


def run(out_dir: Path = CANDIDATE_DIR, seeds=SEEDS) -> Iterator[str]:
    """Rebuild the dataset and train a candidate model (an average of one
    model per seed), yielding log lines."""
    py = sys.executable
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    yield from _stream([py, "scripts/preprocess.py"], ROOT)
    holdout = holdout_signer()
    yield f">>> Test signer (never trained on): {holdout}"
    yield from _stream([py, "scripts/split_dataset.py", "--holdout_signer", holdout], ROOT)
    started = time.time()
    yield f">>> Training {len(seeds)} models side by side (seeds {', '.join(map(str, seeds))})"
    members = [out_dir / f"seed_{seed}" for seed in seeds]
    cmds = {f"seed {seed}": [py, "finetune.py", "--config", str(_write_config(m, seed)), "--arch", "lstm"]
            for seed, m in zip(seeds, members)}
    threads = max(1, (os.cpu_count() or 4) // len(seeds))
    yield from _stream_parallel(cmds, ROOT / "module2_finetune", threads)
    yield ">>> Combining them into one model"
    yield from _stream([py, "scripts/combine_models.py", "--out", str(out_dir), *map(str, members)], ROOT)
    vocab = json.loads((ROOT / "data" / "vocabulary.json").read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(json.dumps({
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "minutes": round((time.time() - started) / 60, 1),
        "test_signer": holdout,
        "words": vocab["metadata"]["total_signs"],
        "clips": int(len(np.load(ROOT / "data" / "processed" / "y.npy"))),
        "models": len(seeds),
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
