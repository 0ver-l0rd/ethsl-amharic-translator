"""
app/ui.py
==========
OpenCV-based display overlay for the real-time EthSL translator.

Renders a sleek HUD showing:
  - Current predicted sign (large Amharic-label display)
  - Confidence bar
  - Frame buffer fill indicator
  - Rolling sentence history
  - FPS counter
  - Speech toggle indicator

Authors: Menase Teshale
"""

import cv2
import numpy as np
from typing import List


# Color palette (BGR)
COLORS = {
    "bg_dark":      (15, 15, 25),
    "accent":       (0, 200, 120),
    "accent2":      (60, 140, 255),
    "warning":      (0, 165, 255),
    "danger":       (0, 60, 200),
    "white":        (255, 255, 255),
    "gray":         (160, 160, 160),
    "black":        (0, 0, 0),
    "panel_bg":     (30, 30, 45),
}


def put_text_shadow(
    frame, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=1.0, color=(255, 255, 255), thickness=2, shadow_color=(0, 0, 0)
):
    """Draw text with a drop shadow for readability."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(frame, pt1, pt2, color, radius=8, thickness=-1, alpha=0.6):
    """Draw a semi-transparent rounded rectangle."""
    overlay = frame.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class SignLanguageUI:
    """
    HUD overlay renderer for the real-time EthSL translator.

    Parameters
    ----------
    vocab : Vocabulary
        Used for display metadata.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def draw(
        self,
        frame: np.ndarray,
        current_label: str,
        current_amharic: str,
        confidence: float,
        sentence_history: List[str],
        buffer_fill: float,
        fps: float,
        speech_enabled: bool,
    ) -> np.ndarray:
        """
        Draw the full HUD overlay on the frame.

        Parameters
        ----------
        frame : np.ndarray  (H, W, 3) BGR
        current_label : str
        current_amharic : str
        confidence : float  (0–1)
        sentence_history : list of Amharic strings
        buffer_fill : float  (0–1), current buffer fill ratio
        fps : float
        speech_enabled : bool

        Returns
        -------
        frame : np.ndarray with HUD drawn
        """
        H, W = frame.shape[:2]

        # ── Top panel ──────────────────────────────────────────────────
        draw_rounded_rect(frame, (0, 0), (W, 90), COLORS["panel_bg"], radius=0, alpha=0.75)

        # Title
        put_text_shadow(frame, "EthSL -> Amharic Translator",
                        (10, 28), scale=0.75, color=COLORS["accent"],
                        thickness=1)

        # FPS
        fps_color = COLORS["accent"] if fps >= 25 else COLORS["warning"]
        put_text_shadow(frame, f"FPS: {fps:.0f}", (W - 110, 28),
                        scale=0.65, color=fps_color, thickness=1)

        # Speech indicator
        speech_text  = "[SPEECH ON]" if speech_enabled else "[SPEECH OFF]"
        speech_color = COLORS["accent"] if speech_enabled else COLORS["gray"]
        put_text_shadow(frame, speech_text, (W - 140, 55),
                        scale=0.5, color=speech_color, thickness=1)

        # ── Buffer fill bar ────────────────────────────────────────────
        bar_y  = 70
        bar_x1, bar_x2 = 10, W - 10
        bar_w  = bar_x2 - bar_x1
        fill_w = int(bar_w * buffer_fill)

        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 12),
                      COLORS["gray"], 1)
        fill_color = COLORS["accent"] if buffer_fill < 0.9 else COLORS["warning"]
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + 12),
                          fill_color, -1)
        put_text_shadow(frame, "Buffer", (bar_x1, bar_y - 3),
                        scale=0.4, color=COLORS["gray"], thickness=1)

        # ── Current prediction panel ────────────────────────────────────
        if current_label:
            pred_y = H - 140
            draw_rounded_rect(frame, (0, pred_y), (W, H), COLORS["panel_bg"],
                               radius=0, alpha=0.80)

            # Sign label
            put_text_shadow(frame, f"Sign: {current_label.replace('_', ' ').upper()}",
                            (12, pred_y + 30), scale=0.8, color=COLORS["white"], thickness=2)

            # Amharic text (displayed as ASCII transliteration since cv2 doesn't support Ethiopic)
            # In production: use PIL for Ethiopic rendering
            put_text_shadow(frame, f"Amharic: {current_amharic}",
                            (12, pred_y + 60), scale=0.9, color=COLORS["accent2"], thickness=2)

            # Confidence bar
            conf_bar_x1, conf_bar_x2 = 12, W - 12
            conf_bar_w = conf_bar_x2 - conf_bar_x1
            conf_fill  = int(conf_bar_w * confidence)
            conf_color = (
                COLORS["accent"]  if confidence >= 0.8 else
                COLORS["warning"] if confidence >= 0.6 else
                COLORS["danger"]
            )
            cv2.rectangle(frame, (conf_bar_x1, pred_y + 75),
                          (conf_bar_x2, pred_y + 88), COLORS["gray"], 1)
            if conf_fill > 0:
                cv2.rectangle(frame, (conf_bar_x1, pred_y + 75),
                              (conf_bar_x1 + conf_fill, pred_y + 88), conf_color, -1)
            put_text_shadow(frame, f"Confidence: {confidence*100:.1f}%",
                            (12, pred_y + 105), scale=0.55, color=conf_color, thickness=1)

            # Sentence history
            history_str = "  |  ".join(sentence_history) if sentence_history else "-"
            put_text_shadow(frame, f"History: {history_str}",
                            (12, pred_y + 125), scale=0.45, color=COLORS["gray"], thickness=1)

        # ── Instructions ───────────────────────────────────────────────
        else:
            # No prediction yet — show instructions at bottom
            put_text_shadow(frame, "Perform a sign in front of the camera",
                            (int(W * 0.15), H - 20), scale=0.6,
                            color=COLORS["gray"], thickness=1)

        # ── Keyboard hints (bottom-left) ───────────────────────────────
        hints = "[Q] Quit  [R] Reset  [S] Speech  [C] Clear"
        put_text_shadow(frame, hints, (10, H - 5),
                        scale=0.4, color=COLORS["gray"], thickness=1)

        return frame
