"""
scripts/combine_models.py
=========================
Combine models trained with different seeds (by finetune.py, one folder each)
into one ensemble model that averages them (src/ensemble.py), and score it
the way finetune.py scores one model:

  - <out>/best_model.h5: the average of every <member>/best_model.h5 (the live model)
  - signer-dependent accuracy: that ensemble on data/splits/test_idx.npy
  - signer-independent accuracy: the average of every member's model trained
    without the held-out signer (<member>/lstm/signer_independent/best.keras),
    on that signer's clips
  - <out>/all_results.json in finetune.py's format, so retrain.scores() reads it

Usage (src/retrain.py runs it):
  python scripts/combine_models.py --out models/candidate models/candidate/seed_42 models/candidate/seed_7 ...
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module2_finetune"))

from src.ensemble import combine, combine_files
from src.vocabulary import Vocabulary


def main():
    parser = argparse.ArgumentParser(description="Combine seed models into one ensemble")
    parser.add_argument("members", nargs="+", help="Folders written by finetune.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--processed_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--splits_dir", default=str(ROOT / "data" / "splits"))
    parser.add_argument("--vocab", default=str(ROOT / "data" / "vocabulary.json"))
    args = parser.parse_args()

    from tensorflow import keras
    from finetune import evaluate_model, measure_latency

    out, members = Path(args.out), [Path(m) for m in args.members]
    labels = Vocabulary(args.vocab).get_all_labels()
    X = np.load(Path(args.processed_dir) / "X.npy").astype(np.float32)
    y = np.load(Path(args.processed_dir) / "y.npy")
    splits = Path(args.splits_dir)

    combine_files([m / "best_model.h5" for m in members], out / "best_model.h5")
    live = keras.models.load_model(str(out / "best_model.h5"), compile=False)
    print(f"Saved the average of {len(members)} models as {out / 'best_model.h5'}")

    results = {"architecture": f"lstm x{len(members)} (ensemble)", "members": len(members),
               "member_folders": [m.name for m in members]}
    test = np.load(splits / "test_idx.npy")
    results["signer_dependent"] = evaluate_model(live, X[test], y[test], labels, "Signer-Dependent")

    si_paths = [m / "lstm" / "signer_independent" / "best.keras" for m in members]
    si_test = splits / "signer_independent_test_idx.npy"
    if all(p.exists() for p in si_paths) and si_test.exists():
        si = combine([keras.models.load_model(str(p), compile=False) for p in si_paths])
        idx = np.load(si_test)
        results["signer_independent"] = evaluate_model(si, X[idx], y[idx], labels, "Signer-Independent")
        singles = [json.loads((m / "all_results.json").read_text(encoding="utf-8"))[0]
                   .get("signer_independent", {}).get("accuracy") for m in members]
        results["signer_independent"]["single_models"] = singles
        print("  One model at a time:", ", ".join(f"{a:.1%}" for a in singles if a is not None))

    results["latency_ms"] = measure_latency(live, X[test][:1])
    print(f"  Latency: {results['latency_ms']:.2f} ms per sign")
    (out / "all_results.json").write_text(json.dumps([results], indent=2, ensure_ascii=False),
                                          encoding="utf-8")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Combined in {time.time() - t:.0f} s")
