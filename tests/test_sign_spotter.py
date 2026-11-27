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
from tests.test_split_recording import hands_up_track, track

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


def test_slow_sign_gets_a_longer_window_with_rest_on_both_sides():
    _, events, _ = run([(2.0, 5.2)])                # 3.2 s, longer than SIGN_WINDOW_SECONDS
    assert len(events) == 1
    # window = 3.2 + 2 x 0.25 = 3.7 s (3.6 +- 1.85); it is already complete when the
    # sign is found to be over (hands down for MIN_REST), so the guess comes then
    assert 5.2 + 0.3 < events[0][0] < 5.2 + 0.5
    hand = np.abs(events[0][1].sequence[:, 132 + 63:]).sum(1) > 0
    assert not hand[0] and not hand[-1]             # the whole sign is inside


def run_up(segments, up, duration, lowering=None):
    """Like run(), for a signer who keeps the hand up and only pauses between signs."""
    times, kp = hands_up_track(segments, up=up, duration=duration, fps=FPS, lowering=lowering)
    spotter, events = SignSpotter(), []
    for t, k in zip(times, kp):
        if (e := spotter.add_frame(float(t), k)):
            events.append((float(t), e))
    return events


def test_signs_with_the_hands_kept_up_are_split_at_pauses():
    # two signs, a still pause between them, hand lowered only at the end
    events = run_up([(1.5, 2.5), (3.2, 4.2)], up=(1.0, 5.5), duration=8.0)
    assert [e.kind for _, e in events] == ["sign", "sign"]
    assert events[0][0] < 3.2 + 0.5        # the first is guessed before the second is over


def test_long_phrase_with_pauses_is_not_too_long():
    events = run_up([(1.3, 2.3), (2.9, 3.9), (4.5, 5.5), (6.1, 7.1)], up=(1.0, 8.0), duration=10.0)
    assert [e.kind for _, e in events] == ["sign"] * 4


def test_lowering_after_a_pause_is_not_a_sign():
    events = run_up([(1.5, 2.5)], up=(1.0, 4.0), duration=6.0, lowering=(3.75, 4.0))
    assert [e.kind for _, e in events] == ["sign"]


def test_moving_without_pausing_is_still_too_long():
    events = run_up([(1.0, 1.0 + MAX_SIGN + 2)], up=(1.0, MAX_SIGN + 4), duration=MAX_SIGN + 6)
    assert [e.kind for _, e in events] == ["too_long"]
