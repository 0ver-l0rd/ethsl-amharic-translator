"""
module1_refinement/download_pretrained.py
==========================================
Downloads pre-built sign language model weights from public sources
and prepares them for refinement.

Strategy: We use the landmark-based LSTM architecture from the
closest open-source repository matching the EthSL pipeline
(MediaPipe + LSTM trained on ASL/general SL datasets).
These weights are then transferred and the final layers replaced
for the EthSL vocabulary.

Run this script ONCE before running refine_model.py.

Authors: Emanuel Solomon
"""

import os
import sys
import json
import hashlib
import argparse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Pretrained model sources
# ---------------------------------------------------------------------------
# We include multiple publicly available sources for sign language models.
# The user can choose which base to start from.

PRETRAINED_SOURCES = {
    "sign_mediapipe_lstm": {
        "description": (
            "LSTM model trained on MediaPipe landmark sequences for "
            "sign language recognition (general SL, skeleton-based). "
            "Architecture: LSTM(128)->LSTM(64)->Dense(60). "
            "Trained on 100-class ASL landmark dataset."
        ),
        "url": "https://github.com/nicholasM95/hand-gesture-recognition-mediapipe/raw/main/model/keypoint_classifier/keypoint_classifier.hdf5",
        "filename": "sign_mediapipe_lstm_base.h5",
        "architecture": "lstm",
        "note": "Replace classification head for EthSL vocabulary"
    },
    "asl_mediapipe_lstm_tf": {
        "description": (
            "TFLite/TF SavedModel LSTM trained on ASL landmarks. "
            "Source: jamesjbustos/sign-language-recognition (GitHub). "
            "Adapted for landmark-based sequence input."
        ),
        "url": "https://raw.githubusercontent.com/jamesjbustos/sign-language-recognition/main/model/sign_language_model.h5",
        "filename": "asl_mediapipe_lstm_tf.h5",
        "architecture": "lstm",
        "note": "Pre-trained on 26 ASL alphabet classes; encoder layers are reused"
    },
}

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "pretrained"

FALLBACK_NOTICE = """
╔══════════════════════════════════════════════════════════════════════════╗
║  PRETRAINED WEIGHTS — FALLBACK NOTICE                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  The requested pretrained weights could not be downloaded automatically. ║
║  This may be because the host URL has changed or is temporarily down.    ║
║                                                                           ║
║  ALTERNATIVES:                                                            ║
║  1. Manually download weights and place in:                               ║
║       models/pretrained/<filename>                                        ║
║                                                                           ║
║  2. Use the scaffold weights (randomly initialized with correct           ║
║       architecture) and train from scratch on your EthSL data.           ║
║       Run: python module1_refinement/refine_model.py --from_scratch       ║
║                                                                           ║
║  3. Request landmark data from prior EthSL research groups               ║
║       (see project proposal §5.2) and train from scratch.                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    """Download a file from URL to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if verbose:
            print(f"  Downloading: {url}")
            print(f"  -> {dest}")

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size / total_size * 100, 100)
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, _progress)
        print()  # newline after progress
        return True
    except Exception as e:
        if verbose:
            print(f"\n  [ERROR] Download failed: {e}")
        return False


def create_scaffold_weights(architecture: str, output_path: Path, num_classes: int = 60) -> None:
    """
    Create randomly-initialized model weights with the correct architecture.
    Used as a fallback when pretrained weights cannot be downloaded.
    """
    print(f"  Creating scaffold weights for architecture: {architecture}")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from model_architectures import build_model

        model = build_model(architecture, num_classes=num_classes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        print(f"  Scaffold weights saved to: {output_path}")
        print(f"  Model summary:")
        model.summary()
    except ImportError as e:
        print(f"  [ERROR] Could not build scaffold model: {e}")
        print("  Install TensorFlow first: pip install tensorflow")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained sign language model weights"
    )
    parser.add_argument(
        "--model",
        choices=list(PRETRAINED_SOURCES.keys()) + ["all"],
        default="sign_mediapipe_lstm",
        help="Which pretrained model to download",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=60,
        help="Number of output classes for scaffold weights",
    )
    parser.add_argument(
        "--scaffold_only",
        action="store_true",
        help="Skip download, create scaffold (random init) weights only",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models_to_process = (
        list(PRETRAINED_SOURCES.keys()) if args.model == "all"
        else [args.model]
    )

    for model_key in models_to_process:
        info = PRETRAINED_SOURCES[model_key]
        dest = OUTPUT_DIR / info["filename"]

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"Description: {info['description']}")
        print(f"Note: {info['note']}")
        print(f"{'='*60}")

        if dest.exists():
            print(f"  Already downloaded: {dest}")
            continue

        if args.scaffold_only:
            scaffold_path = OUTPUT_DIR / f"{model_key}_scaffold.h5"
            create_scaffold_weights(info["architecture"], scaffold_path, args.num_classes)
            continue

        success = download_file(info["url"], dest)

        if not success:
            print(FALLBACK_NOTICE)
            scaffold_path = OUTPUT_DIR / f"{model_key}_scaffold.h5"
            print(f"\nCreating scaffold weights as fallback...")
            create_scaffold_weights(info["architecture"], scaffold_path, args.num_classes)

    # Save metadata
    meta = {
        "sources": PRETRAINED_SOURCES,
        "output_dir": str(OUTPUT_DIR),
        "usage": "Use refine_model.py to adapt these weights to EthSL vocabulary"
    }
    with open(OUTPUT_DIR / "pretrained_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Pretrained weights directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
