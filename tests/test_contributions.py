"""
tests/test_contributions.py
===========================
Tests for src/contributions.py: uploads are split into repetitions, wait for
review, and only approved repetitions reach the training data.

Run with: pytest tests/
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import contributions as c
from src.vocabulary import Vocabulary, add_word
from tests.test_split_recording import track

SIGNS = [(2.0, 3.4), (5.0, 6.3), (8.0, 9.5)]
DURATION = 11.0


def fake_video(path, duration=DURATION, fps=30):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(duration * fps)):
        out.write(np.full((48, 64, 3), i % 255, np.uint8))
    out.release()
    return path


def fake_analyse(video):
    times, kp = track(SIGNS, duration=DURATION)
    return times, kp, 30.0, DURATION


@pytest.fixture
def env(tmp_path):
    vocab = tmp_path / "vocabulary.json"
    vocab.write_text(json.dumps({"metadata": {"total_signs": 1, "categories": ["social"]},
                                 "signs": {"social": [{"label": "misgana", "amharic": "ምስጋና",
                                                       "description": "Thanks", "id": 0}]}},
                                ensure_ascii=False), encoding="utf-8")
    return {"root": tmp_path / "contributions", "landmarks": tmp_path / "landmarks",
            "vocab": vocab, "video": fake_video(tmp_path / "upload.mp4")}


def upload(env, name="Abebe Kebede", label="selam", amharic="ሰላም", new=True):
    return c.split_upload(env["video"], label, amharic, "Hello", new, name,
                          root=env["root"], analyse=fake_analyse)


def fake_extract(clips):
    for p in clips:
        np.save(p.with_name(p.stem + "_kp.npy"), np.zeros((30, 258), np.float32))


def test_upload_is_split_into_every_repetition_and_waits_for_review(env):
    u = upload(env)
    assert len(u.reps) == 3 and u.status == "pending"
    assert all(p.exists() and p.stat().st_size > 0 for p in u.clip_paths(env["root"]))
    assert [x.id for x in c.list_uploads("pending", env["root"])] == [u.id]
    assert not env["landmarks"].exists()            # nothing reaches training yet


def test_contributors_get_anonymous_ids_and_keep_them(env):
    root = env["root"]
    assert c.signer_id_for("Abebe Kebede", root) == "c01"
    assert c.signer_id_for("  abebe   KEBEDE ", root) == "c01"
    assert c.signer_id_for("Sara", root) == "c02"
    assert c.signer_id_for("Estifanos Behailu", root) == "s01"
    assert c.signer_id_for("Estifanos", root, team=False) == "c03"   # public page


def test_approve_adds_only_kept_reps_and_the_new_word(env):
    u = upload(env)
    done = c.approve(u.id, keep=[1, 3], root=env["root"], landmarks_dir=env["landmarks"],
                     vocab_path=env["vocab"], extract=fake_extract)
    assert done.status == "approved"
    files = sorted(p.name for p in (env["landmarks"] / "selam").iterdir())
    assert files == ["c01_seq_000.mp4", "c01_seq_000_kp.npy", "c01_seq_001.mp4", "c01_seq_001_kp.npy"]
    vocab = Vocabulary(str(env["vocab"]))
    assert vocab.label_to_amharic("selam") == "ሰላም" and vocab.label_to_id("selam") == 1
    assert vocab.label_to_id("misgana") == 0        # existing IDs never move


def test_second_upload_appends_after_the_first(env):
    for _ in range(2):
        u = upload(env)
        c.approve(u.id, keep=[1, 2, 3], root=env["root"], landmarks_dir=env["landmarks"],
                  vocab_path=env["vocab"], extract=fake_extract)
    assert len(list((env["landmarks"] / "selam").glob("c01_seq_*_kp.npy"))) == 6
    assert Vocabulary(str(env["vocab"])).num_classes == 2


def test_reject_deletes_the_clips(env):
    u = upload(env)
    c.reject(u.id, env["root"])
    assert c.load(u.id, env["root"]).status == "rejected"
    assert not any(p.exists() for p in u.clip_paths(env["root"]))
    with pytest.raises(ValueError):
        c.approve(u.id, keep=[1], root=env["root"], landmarks_dir=env["landmarks"],
                  vocab_path=env["vocab"], extract=fake_extract)


def test_approve_needs_at_least_one_rep(env):
    u = upload(env)
    with pytest.raises(ValueError):
        c.approve(u.id, keep=[], root=env["root"], landmarks_dir=env["landmarks"],
                  vocab_path=env["vocab"], extract=fake_extract)


def test_add_word_rules(env):
    with pytest.raises(ValueError):
        add_word(env["vocab"], "Selam!", "ሰላም")         # not a folder-safe label
    with pytest.raises(ValueError):
        add_word(env["vocab"], "amesegnalehu", "ምስጋና")  # Amharic already there
    with pytest.raises(ValueError):
        add_word(env["vocab"], "misgana", "ሌላ")          # label already used
    assert add_word(env["vocab"], "misgana", "ምስጋና") == 0
