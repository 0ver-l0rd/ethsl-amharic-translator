"""
src/phrases.py
==============
Turns signs signed one after another into an Amharic sentence, for the demo:
እንዴት + አንተ -> "እንዴት ነህ?". The table is data/phrases.json.

  - a phrase's signs may come in any order, as EthSL word order is not
    settled for these phrases
  - each sign must follow the one before within PHRASE_GAP_SECONDS, so two
    words signed minutes apart never join
  - the longest phrase wins, and a longer phrase can take over a shorter one
    just made: እኔ + ደህና -> "ደህና ነኝ", then + ምስጋና -> "ደህና ነኝ፣ አመሰግናለሁ"
  - words that are no phrase are shown on their own
  - for speech, open_from() tells which recent words could still become (a
    longer) phrase, so the app can wait before saying them
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

PHRASES_PATH = Path(__file__).parent.parent / "data" / "phrases.json"
PHRASE_GAP_SECONDS = 6.0
MAX_WORDS = 50        # words remembered for the sentence


@dataclass(frozen=True)
class Phrase:
    signs: tuple          # Amharic words, as in vocabulary.json
    amharic: str
    english: str = ""


@dataclass
class _Word:
    amharic: str
    t: float
    group: int            # words made into one phrase share a group number
    seq: int              # counts up with every word added
    phrase: Optional[Phrase] = None


def load_phrases(path: Path = PHRASES_PATH, vocabulary: Optional[Iterable[str]] = None) -> List[Phrase]:
    """Read the phrase table. With `vocabulary` (Amharic words), unknown signs raise ValueError."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    phrases = [Phrase(tuple(p["signs"]), p["amharic"], p.get("english", "")) for p in data["phrases"]]
    known = set(vocabulary) if vocabulary is not None else None
    for p in phrases:
        if len(p.signs) < 2:
            raise ValueError(f"Phrase '{p.amharic}' needs at least two signs")
        if known is not None and (unknown := [s for s in p.signs if s not in known]):
            raise ValueError(f"Phrase '{p.amharic}' uses signs not in the vocabulary: {unknown}")
    return phrases


def phrases_for(vocab, path: Path = PHRASES_PATH) -> List[Phrase]:
    """The phrase table checked against a Vocabulary; a broken table is reported and left out."""
    try:
        return load_phrases(path, [vocab.label_to_amharic(label) for label in vocab.labels])
    except (OSError, ValueError, KeyError) as e:
        print(f"WARNING: phrases left out ({path.name}): {e}")
        return []


class PhraseBuilder:
    """Collects recognised words and joins them into phrases."""

    def __init__(self, phrases: List[Phrase], gap_seconds: float = PHRASE_GAP_SECONDS):
        self.gap = gap_seconds
        self.longest = max((len(p.signs) for p in phrases), default=0)
        self._table = {}
        self._counts = [Counter(p.signs) for p in phrases]
        for p in phrases:   # the first phrase listed wins if two use the same signs
            self._table.setdefault(frozenset(Counter(p.signs).items()), p)
        self._words: List[_Word] = []
        self._next_group = 0
        self._seq = 0
        self.joined: List[str] = []   # the last phrase's signs, in the order signed

    def add(self, amharic: str, t: float) -> Optional[Phrase]:
        """Add a recognised word. Returns the phrase it completes, or None."""
        self._seq += 1
        self._words = (self._words + [_Word(amharic, t, self._new_group(), self._seq)])[-MAX_WORDS:]
        for n in range(min(self.longest, len(self._words)), 1, -1):
            tail = self._words[-n:]
            if not self._joinable(tail):
                continue
            phrase = self._table.get(frozenset(Counter(w.amharic for w in tail).items()))
            if phrase:
                group = self._new_group()
                for w in tail:
                    w.group, w.phrase = group, phrase
                self.joined = [w.amharic for w in tail]
                return phrase
        return None

    @property
    def last_seq(self) -> int:
        """Number of the newest word (0 before the first)."""
        return self._seq

    def open_from(self) -> Optional[int]:
        """
        Number of the first of the newest words that could still become a
        phrase, or a longer one, if more signs follow; None if they can't.
        """
        for n in range(min(self.longest - 1, len(self._words)), 0, -1):
            tail = self._words[-n:]
            if not self._joinable(tail):
                continue
            have = Counter(w.amharic for w in tail)
            if any(not have - c and sum(c.values()) > n for c in self._counts):   # part of a longer one
                return tail[0].seq
        return None

    def pieces(self, after: int, before: Optional[int] = None) -> List[str]:
        """The sentence's words and phrases made of words numbered after < seq < before."""
        out, last = [], None
        for w in self._words:
            if w.seq > after and (before is None or w.seq < before):
                if w.group != last:
                    out.append(w.phrase.amharic if w.phrase else w.amharic)
                last = w.group
        return out

    def _joinable(self, tail: List[_Word]) -> bool:
        """Close enough in time, and not splitting a phrase already made."""
        if any(b.t - a.t > self.gap for a, b in zip(tail, tail[1:])):
            return False
        before = len(self._words) - len(tail)
        return before == 0 or self._words[before - 1].group != tail[0].group

    def _new_group(self) -> int:
        self._next_group += 1
        return self._next_group

    def sentence(self) -> List[str]:
        """What was said so far: phrases in place of their signs, other words as they are."""
        out, last = [], None
        for w in self._words:
            if w.group != last:
                out.append(w.phrase.amharic if w.phrase else w.amharic)
            last = w.group
        return out

    def clear(self) -> None:
        self._words, self.joined = [], []
