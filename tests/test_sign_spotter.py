"""
tests/test_sign_spotter.py
==========================
Tests for src/sign_spotter.py: the live app classifies once per sign, and
never while the hands are down.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sign_spotter import MAX_SIGN, SignSpotter, Status
from tests.test_split_recording import track

FPS = 15   # the live app runs slower than the camera


def run(segments, duration=12.0, body=True):
    """Feed a signer doing `segments` to a spotter. Returns (spotter, events with times, statuses)."""
    times, kp = track(segments, duration=duration, fps=FPS)
    if not body:
        kp[:] = 0
    spotter, events, statuses = SignSpotter(), [], []
    for t, k in zip(times, kp):
        e = spotter.add_frame(float(t), k)
        if e:
            events.append((float(t), e))
        statuses.append(spotter.status)
    return spotter, events, statuses


def test_no_guess_while_hands_are_down():
    _, events, statuses = run([])
    assert events == []
    assert set(statuses[FPS:]) == {Status.READY}


def test_one_guess_per_sign():
    _, events, _ = run([(2.0, 3.5), (6.0, 7.2)])
    assert [e.kind for _, e in events] == ["sign", "sign"]
    # Each guess comes when the window centred on the sign is complete
    assert abs(events[0][0] - (2.75 + 1.25)) < 0.15
    assert all(e.sequence.shape == (30, 258) for _, e in events)


def test_window_is_centred_on_the_sign():
    _, events, _ = run([(2.0, 3.5)])
    seq = events[0][1].sequence
    hand = np.abs(seq[:, 132 + 63:]).sum(1) > 0
    first, last = np.flatnonzero(hand)[[0, -1]]
    assert abs(first - (29 - last)) <= 1        # same rest before and after
    assert not hand[:3].any() and not hand[-3:].any()


def test_twitch_is_ignored():
    _, events, _ = run([(3.0, 3.2)])
    assert events == []


def test_short_dip_does_not_split_a_sign():
    _, events, _ = run([(2.0, 2.8), (2.95, 3.6)])
    assert [e.kind for _, e in events] == ["sign"]


def test_hands_up_too_long_warns_once_and_is_not_classified():
    _, events, _ = run([(1.0, 1.0 + MAX_SIGN + 3)])
    assert [e.kind for _, e in events] == ["too_long"]


def test_status_follows_the_sign():
    _, _, statuses = run([(2.0, 3.5)], duration=6.0)
    at = lambda s: statuses[int(s * FPS)]
    assert at(1.0) == Status.READY
    assert at(3.0) == Status.SIGNING
    assert at(3.9) == Status.RECOGNISING     # hands down 0.35 s, window not full yet
    assert at(5.0) == Status.READY


def test_nobody_in_view():
    _, events, statuses = run([], duration=3.0, body=False)
    assert events == [] and statuses[-1] == Status.NO_BODY


def test_reset_drops_the_sign_in_progress():
    times, kp = track([(1.0, 2.0)], duration=5.0, fps=FPS)
    spotter, kinds = SignSpotter(), []
    for t, k in zip(times, kp):
        if abs(t - 2.2) < 1 / FPS / 2:
            spotter.reset()
        e = spotter.add_frame(float(t), k)
        if e:
            kinds.append(e.kind)
    assert kinds == []
