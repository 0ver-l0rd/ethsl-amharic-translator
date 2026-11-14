"""
app/web_app.py
==============
The translator in a web browser: live translation plus adding training
videos, without the command line.

  Translate    Sign in front of the webcam; each sign is shown in Amharic
               and spoken (the same logic as app/realtime_app.py).
  Add a video  Upload (or record with the webcam) one word signed any number
               of times, hands down between repetitions. It is split into one
               clip per repetition and waits for review.
  Review       (team) Watch each upload's clips, untick bad ones, approve or
               reject. Only approved clips are used for training.
  Retrain      (team) Train a new model on all approved clips and compare it
               with the current one; "Use this model" switches the live model.

Usage:
  python app/web_app.py              # team page: all four tabs
  python app/web_app.py --public     # Translate and Add a video only (for hosting)

Then open http://127.0.0.1:7860 in a browser.

Authors: Estifanos Behailu
"""

import argparse
import base64
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import contributions as contrib
from src import retrain
from src.clip_naming import parse_signer
from src.amharic_tts import AmharicTTS
from src.sign_spotter import Status
from src.vocabulary import LABEL_RE, Vocabulary
from app.live_session import LiveSession

NEW_WORD = "+ A new word"

HOW_TO = """
### How to record
1. **Landscape**, with your head, shoulders and both hands in view. Plain background, good light.
2. Sign **one word**, as many times as you like (10-15 is ideal).
3. Between signs, put your **hands fully down for about a second**: that is how the video is split.
4. Pick the word below, upload the video (or record it with your webcam), and press **Submit**.

A team member checks every clip before it is used for training.
"""


def word_choices():
    vocab = Vocabulary(str(contrib.VOCAB_PATH))
    words = [f"{vocab.id_to_amharic(i)}  ({vocab.id_to_label(i)})" for i in range(vocab.num_classes)]
    return words + [NEW_WORD]


def parse_choice(choice: str):
    """'ሰላም  (selam)' -> ('selam', 'ሰላም')"""
    amharic, label = choice.rsplit("(", 1)
    return label.rstrip(")").strip(), amharic.strip()


def caption(n, rep):
    flags = rep.flags
    return f"{n}" + (f"  ⚠ {', '.join(flags)}" if flags else "")


# ── Translate ───────────────────────────────────────────────────────────────

FRAME_EVERY = 0.07            # s between webcam frames sent to the server (~14 fps)
IDLE_SECONDS = 120            # a session with no frames for this long is closed
SPEECH_DIR = Path(tempfile.gettempdir()) / "ethsl_speech"

STATUS_LOOK = {               # text, colour
    Status.NO_BODY:     ("Step into view of the camera", "#6b7280"),
    Status.READY:       ("READY - raise your hands and sign", "#16a34a"),
    Status.SIGNING:     ("SIGNING... lower your hands when done", "#2563eb"),
    Status.RECOGNISING: ("RECOGNISING...", "#d97706"),
}

_engine = None
_engine_lock = threading.Lock()
_sessions = {}                # browser session -> [LiveSession, LandmarkExtractor, lock, last used]
_tts = AmharicTTS()


def get_engine():
    """The live model, loaded once (and again after 'Use this model')."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from app.inference_engine import InferenceEngine
            _engine = InferenceEngine(str(retrain.LIVE_DIR / "best_model.h5"),
                                      Vocabulary(str(contrib.VOCAB_PATH)))
        return _engine


def _close_idle(now):
    for key in [k for k, s in _sessions.items() if now - s[3] > IDLE_SECONDS and not s[2].locked()]:
        _sessions.pop(key)[1].close()


def session_for(request: gr.Request):
    from src.landmark_extractor import LandmarkExtractor, face_mode_for_dim
    now = time.time()
    key = request.session_hash
    if key not in _sessions:
        _close_idle(now)
        engine = get_engine()
        _, seq_len, dim = engine.model.input_shape
        _sessions[key] = [LiveSession(engine, sequence_length=seq_len),
                          LandmarkExtractor(face_mode=face_mode_for_dim(dim)), threading.Lock(), now]
    _sessions[key][3] = now
    return _sessions[key]


def status_html(s: LiveSession) -> str:
    text, colour = STATUS_LOOK[s.status]
    if s.message:
        text, colour = s.message, "#d97706"
    return (f'<div style="background:{colour};color:#fff;border-radius:10px;padding:12px 16px;'
            f'font-size:1.15rem;font-weight:600">{text}</div>')


def word_html(s: LiveSession) -> str:
    if not s.label:
        return ('<div style="padding:28px 8px;font-size:1.1rem;opacity:.75">'
                'Sign one word, then lower your hands. The word appears here.</div>')
    pct = round(s.confidence * 100)
    bar = "#16a34a" if s.confidence >= 0.8 else "#d97706"
    return (f'<div style="padding:8px 4px"><div style="font-size:4rem;line-height:1.2">{s.amharic}</div>'
            f'<div style="opacity:.7;margin-bottom:8px">{s.label.replace("_", " ")}</div>'
            f'<div style="background:rgba(128,128,128,.25);border-radius:6px;height:12px">'
            f'<div style="background:{bar};width:{pct}%;height:12px;border-radius:6px"></div></div>'
            f'<div style="margin-top:4px">{pct}% sure</div></div>')


def history_md(s: LiveSession) -> str:
    return "**History:** " + ("  ·  ".join(s.history[-10:]) if s.history else "-")


def speech_html(word: str) -> str:
    """An autoplaying <audio> with the word's speech inlined (new element each time, so repeats play)."""
    path = _tts.to_file(word, str(SPEECH_DIR))
    if not path:
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<audio autoplay data-t="{time.time()}" src="data:audio/mpeg;base64,{data}"></audio>'


def live_frame(frame, speech, request: gr.Request):
    if frame is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    session, extractor, lock, _ = session_for(request)
    if not lock.acquire(blocking=False):      # still busy with the last frame: drop this one
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    try:
        t = time.perf_counter()
        keypoints, _ = extractor.process_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        word = session.update(t, keypoints)
    finally:
        lock.release()
    audio = speech_html(word) if word and speech else gr.skip()
    return status_html(session), word_html(session), history_md(session), audio


def live_reset(request: gr.Request):
    s = session_for(request)[0]
    s.reset()
    return status_html(s), word_html(s), history_md(s)


def live_clear(request: gr.Request):
    s = session_for(request)[0]
    s.clear_history()
    return history_md(s)


# ── Add a video ─────────────────────────────────────────────────────────────

def submit(name, choice, amharic, label, meaning, video, consent, public):
    if not consent:
        raise gr.Error("Please tick the consent box first.")
    if not video:
        raise gr.Error("Please upload or record a video.")
    if not choice:
        raise gr.Error("Please pick the word you are signing.")
    if choice == NEW_WORD:
        label = (label or "").strip().lower()
        if not (amharic or "").strip():
            raise gr.Error("Type the new word in Amharic.")
        if not LABEL_RE.match(label):
            raise gr.Error("The name in Latin letters must be lower case, e.g. selam or dehna_neh.")
        if label in Vocabulary(str(contrib.VOCAB_PATH)).labels:
            raise gr.Error(f"'{label}' is already a word: pick it from the list instead.")
        new = True
    else:
        (label, amharic), meaning, new = parse_choice(choice), "", False
    try:
        upload = contrib.split_upload(Path(video), label, amharic, meaning or "", new, name,
                                      team=not public)
    except ValueError as e:
        raise gr.Error(str(e))
    if not upload.reps:
        return ("**No signs found.** Make sure your hands are in view while signing and fully "
                "down between signs, then try again."), []
    clips = [(str(p), caption(n, r)) for n, (p, r) in enumerate(zip(upload.clip_paths(), upload.reps), 1)]
    return (f"**Thank you!** Found **{len(upload.reps)}** repetitions of {amharic}. "
            f"A team member will review them before they are used."), clips


# ── Review ──────────────────────────────────────────────────────────────────

def pending_choices():
    return [f"{u.id}  ·  {u.amharic} ({u.label}){' NEW WORD' if u.new_word else ''}  ·  "
            f"{u.signer}  ·  {len(u.reps)} reps" for u in contrib.list_uploads("pending")]


def refresh_pending():
    choices = pending_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def show_upload(choice):
    if not choice:
        return [], gr.update(choices=[], value=[]), "No uploads waiting for review."
    u = contrib.load(choice.split()[0])
    clips = [(str(p), caption(n, r)) for n, (p, r) in enumerate(zip(u.clip_paths(), u.reps), 1)]
    numbers = [str(n) for n in range(1, len(u.reps) + 1)]
    unflagged = [str(n) for n, r in enumerate(u.reps, 1) if not r.flags]
    info = (f"**{u.amharic}** ({u.label}{', a new word' if u.new_word else ''})"
            f"{' - ' + u.description if u.description else ''} · signer **{u.signer}** · "
            f"{len(u.reps)} repetitions. Flagged clips start unticked.")
    return clips, gr.update(choices=numbers, value=unflagged), info


def approve(choice, keep):
    if not choice:
        raise gr.Error("Pick an upload first.")
    try:
        u = contrib.approve(choice.split()[0], [int(k) for k in keep])
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info(f"Added {len(u.kept_clips)} clips of {u.amharic} for {u.signer}.")
    return refresh_pending()


def reject(choice):
    if not choice:
        raise gr.Error("Pick an upload first.")
    contrib.reject(choice.split()[0])
    gr.Info("Rejected; its clips were deleted.")
    return refresh_pending()


# ── Retrain ─────────────────────────────────────────────────────────────────

def data_summary():
    vocab = Vocabulary(str(contrib.VOCAB_PATH))
    per_word, signers = Counter(), set()
    for label in vocab.labels:
        for kp in (contrib.LANDMARKS_DIR / label).glob("*_kp.npy"):
            per_word[label] += 1
            signers.add(parse_signer(kp))
    ours = sorted(s for s in signers if s and not s.startswith("g"))
    thin = [f"{vocab.label_to_amharic(l)} ({per_word[l]})" for l in vocab.labels if per_word[l] < 30]
    lines = [f"**{sum(per_word.values())}** clips · **{vocab.num_classes}** words · "
             f"**{len(signers)}** signers (ours: {', '.join(ours) or 'none yet'})",
             f"Waiting for review: **{len(contrib.list_uploads('pending'))}** uploads"]
    if thin:
        lines.append(f"Words with fewer than 30 clips (weak until more arrive): {', '.join(thin)}")
    return "\n\n".join(lines)


def scores_table():
    def fmt(s, key):
        v = (s or {}).get(key)
        return "-" if v is None else (f"{v:.1%}" if isinstance(v, float) else str(v))
    cur, new = retrain.scores(retrain.LIVE_DIR), retrain.scores(retrain.CANDIDATE_DIR)
    rows = [("Accuracy on the test signer (never trained on)", "unseen_signer"),
            ("Accuracy on signers it trained on", "seen_signers"),
            ("Test signer", "test_signer"), ("Words", "words"), ("Clips", "clips"),
            ("Trained", "trained")]
    md = "| | Current model | New model |\n|---|---|---|\n"
    md += "\n".join(f"| {name} | {fmt(cur, k)} | {fmt(new, k)} |" for name, k in rows)
    if new and cur and new.get("test_signer") != cur.get("test_signer"):
        md += "\n\nThe two were tested on different signers, so compare with care."
    return md


def run_retrain():
    log = []
    try:
        for line in retrain.run():
            if "━" in line:          # Keras progress bars
                continue
            log.append(line)
            yield "\n".join(log[-300:]), scores_table()
    except RuntimeError as e:
        log.append(f"FAILED: {e}")
    yield "\n".join(log[-300:]), scores_table()


def use_new_model():
    global _engine
    try:
        retrain.promote()
    except FileNotFoundError as e:
        raise gr.Error(str(e))
    with _engine_lock:
        _engine = None
    for key in list(_sessions):              # sessions hold the old model
        _sessions.pop(key)[1].close()
    gr.Info("Switched: Translate uses the new model now, and the desktop app from its next start.")
    return scores_table()


# ── Page ────────────────────────────────────────────────────────────────────

# Hide Gradio's footer ("Use via API", "Built with Gradio")
CSS = "footer { display: none !important; }"


def build(public: bool) -> gr.Blocks:
    with gr.Blocks(title="EthSL → Amharic Translator", css=CSS) as page:
        gr.Markdown("# EthSL → Amharic Translator")
        with gr.Tab("Translate"):
            with gr.Row():
                with gr.Column(scale=3):
                    cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera",
                                   mirror_webcam=True)
                with gr.Column(scale=2):
                    blank = LiveSession(None)
                    status = gr.HTML(status_html(blank))
                    word_box = gr.HTML(word_html(blank))
                    history = gr.Markdown(history_md(blank))
                    speech = gr.Checkbox(value=True, label="Speak each word")
                    with gr.Row():
                        reset_btn = gr.Button("Reset")
                        clear_btn = gr.Button("Clear history")
                    speaker = gr.HTML(visible=True, elem_id="speaker")
                    gr.Markdown("Press the camera button to start. Sign one word, then put your hands "
                                "down; the word appears when you lower them.")
            cam.stream(live_frame, [cam, speech], [status, word_box, history, speaker],
                       stream_every=FRAME_EVERY, time_limit=None, concurrency_limit=None,
                       show_progress="hidden")
            reset_btn.click(live_reset, None, [status, word_box, history])
            clear_btn.click(live_clear, None, history)

        with gr.Tab("Add a video"):
            gr.Markdown(HOW_TO)
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="Your name",
                                      info="Kept private: the clips are stored under an ID like c01.")
                    word = gr.Dropdown(word_choices(), label="Word you are signing", value=None)
                    with gr.Group(visible=False) as new_box:
                        amharic = gr.Textbox(label="New word in Amharic", placeholder="ሰላም")
                        label = gr.Textbox(label="Name in Latin letters", placeholder="selam")
                        meaning = gr.Textbox(label="Meaning in English (optional)", placeholder="Hello")
                    consent = gr.Checkbox(label="I agree that the EthSL team may store my clips and use "
                                                "them to train this model.")
                with gr.Column():
                    video = gr.Video(label="Your video", sources=["upload", "webcam"])
                    send = gr.Button("Submit", variant="primary")
            result = gr.Markdown()
            gallery = gr.Gallery(label="Repetitions found", columns=5, height="auto")
            word.change(lambda c: gr.update(visible=c == NEW_WORD), word, new_box)
            send.click(lambda *a: submit(*a, public), [name, word, amharic, label, meaning, video, consent],
                       [result, gallery])

        if not public:
            with gr.Tab("Review") as review_tab:
                with gr.Row():
                    pending = gr.Dropdown(pending_choices(), label="Uploads waiting for review", scale=4)
                    refresh = gr.Button("Refresh", scale=1)
                info = gr.Markdown()
                clips = gr.Gallery(label="Repetitions", columns=5, height="auto")
                keep = gr.CheckboxGroup(label="Keep these repetitions")
                with gr.Row():
                    ok = gr.Button("Approve kept repetitions", variant="primary")
                    no = gr.Button("Reject whole upload", variant="stop")
                pending.change(show_upload, pending, [clips, keep, info])
                refresh.click(refresh_pending, None, pending)
                review_tab.select(refresh_pending, None, pending)
                ok.click(approve, [pending, keep], pending)
                no.click(reject, pending, pending)

            with gr.Tab("Retrain") as train_tab:
                summary = gr.Markdown(data_summary())
                table = gr.Markdown(scores_table())
                with gr.Row():
                    go = gr.Button("Retrain on all approved clips", variant="primary")
                    use = gr.Button("Use this model")
                gr.Markdown("Training takes several minutes. The live app keeps the current model "
                            "until you press **Use this model** (the old one is kept as previous_model.h5).")
                log = gr.Textbox(label="Training log", lines=15, max_lines=15, autoscroll=True)
                train_tab.select(lambda: (data_summary(), scores_table()), None, [summary, table])
                go.click(run_retrain, None, [log, table])
                use.click(use_new_model, None, table)
        # Words added since the page started (approved new words) show up on reload
        page.load(lambda: gr.update(choices=word_choices()), None, word)
    return page


def main():
    parser = argparse.ArgumentParser(description="EthSL translator in the browser")
    parser.add_argument("--public", action="store_true",
                        help="Only Translate and Add a video, and team names aren't recognised")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build(args.public).queue().launch(server_port=args.port, show_api=False,
                                      allowed_paths=[str(contrib.CONTRIB_DIR)])


if __name__ == "__main__":
    main()
