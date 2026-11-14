"""
src/vocabulary.py
=================
Utilities for loading and using the EthSL sign vocabulary.

Provides a Vocabulary class that loads vocabulary.json and exposes
label <=> Amharic text mapping functions.

Authors: Estifanos Behailu, Emanuel Solomon
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VOCABULARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "vocabulary.json"
)


LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")   # folder-safe Latin label, e.g. selam, dehna_neh


def add_word(vocab_path, label: str, amharic: str, description: str = "",
             category: str = "everyday") -> int:
    """
    Add a sign to vocabulary.json with the next free ID and return the ID.
    If the label is already there with the same Amharic, its ID is returned
    unchanged. New signs always go last, so existing IDs (the model's output
    order) never move.
    """
    if not LABEL_RE.match(label):
        raise ValueError(f"Label '{label}' must be lower-case Latin letters, digits or _ "
                         f"(starting with a letter), e.g. selam or dehna_neh.")
    if not amharic.strip():
        raise ValueError("The Amharic word is empty.")
    path = Path(vocab_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    signs = [s for group in data["signs"].values() for s in group]
    for s in signs:
        if s["label"] == label:
            if s["amharic"] != amharic.strip():
                raise ValueError(f"Label '{label}' is already used for {s['amharic']}.")
            return s["id"]
        if s["amharic"] == amharic.strip():
            raise ValueError(f"{amharic} is already in the vocabulary as '{s['label']}'.")
    new_id = len(signs)
    data["signs"].setdefault(category, []).append(
        {"label": label, "amharic": amharic.strip(), "description": description.strip(), "id": new_id})
    data["metadata"]["total_signs"] = new_id + 1
    if category not in data["metadata"].setdefault("categories", []):
        data["metadata"]["categories"].append(category)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return new_id


class Vocabulary:
    """
    Loads EthSL vocabulary and provides label <=> index <=> Amharic mapping.

    Parameters
    ----------
    vocab_path : str
        Path to vocabulary.json file.
    """

    def __init__(self, vocab_path: str = VOCABULARY_PATH):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._signs: List[Dict] = []
        for category, signs in data["signs"].items():
            for sign in signs:
                sign["category"] = category
                self._signs.append(sign)

        # Build lookup tables
        self._id_to_label:   Dict[int, str] = {s["id"]: s["label"]   for s in self._signs}
        self._id_to_amharic: Dict[int, str] = {s["id"]: s["amharic"] for s in self._signs}
        self._label_to_id:   Dict[str, int] = {s["label"]: s["id"]   for s in self._signs}
        self._label_to_amharic: Dict[str, str] = {s["label"]: s["amharic"] for s in self._signs}

        self.num_classes = len(self._signs)
        self.labels: List[str] = [self._id_to_label[i] for i in range(self.num_classes)]

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def label_to_amharic(self, label: str) -> str:
        """Convert a sign label (string) to its Amharic text."""
        return self._label_to_amharic.get(label, label)

    def id_to_amharic(self, class_id: int) -> str:
        """Convert a class index to its Amharic text."""
        return self._id_to_amharic.get(class_id, str(class_id))

    def id_to_label(self, class_id: int) -> str:
        """Convert a class index to its sign label."""
        return self._id_to_label.get(class_id, str(class_id))

    def label_to_id(self, label: str) -> Optional[int]:
        """Convert a sign label to its class index."""
        return self._label_to_id.get(label, None)

    def get_all_labels(self) -> List[str]:
        """Return list of all sign labels in order of class index."""
        return self.labels

    def get_categories(self) -> List[str]:
        """Return list of unique category names."""
        return list({s["category"] for s in self._signs})

    def describe(self, label: str) -> Optional[str]:
        """Return the English description for a sign label."""
        for s in self._signs:
            if s["label"] == label:
                return s.get("description", "")
        return None

    def top_k_prediction(
        self, probabilities: "np.ndarray", k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Convert a softmax probability array into top-k predictions.

        Parameters
        ----------
        probabilities : np.ndarray of shape (num_classes,)
        k : int

        Returns
        -------
        list of (label, amharic, confidence) tuples sorted by confidence desc
        """
        import numpy as np
        top_indices = np.argsort(probabilities)[::-1][:k]
        return [
            (self.id_to_label(i), self.id_to_amharic(i), float(probabilities[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return self.num_classes

    def __repr__(self) -> str:
        return f"Vocabulary(num_classes={self.num_classes})"
