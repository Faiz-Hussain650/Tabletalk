"""Speech-to-text with mlx-whisper (runs on the Apple Silicon GPU).

The menu prompt trick: Whisper accepts an `initial_prompt`. Listing menu words there
makes it spell "margherita", "bruschetta" and "tiramisu" correctly far more often.
"""

from __future__ import annotations

import re
import subprocess
import time

import numpy as np

from agent.order_state import Menu

SAMPLE_RATE = 16000


def menu_prompt(menu: Menu) -> str:
    names = [i.name for i in menu.items]
    return f"Phone order at an Italian restaurant. Menu: {', '.join(names)}."


# Whisper "hears" these in silence or noise. Treat them as nothing.
HALLUCINATIONS = {"you", "thank you", "thank you.", "thanks for watching", "thanks for watching!", ".", "bye"}
NON_LATIN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]")


def is_speech(audio: np.ndarray, min_seconds: float = 0.4, min_rms: float = 0.004) -> bool:
    """Skip clips that are too short or too quiet - that's where Whisper invents words."""
    if len(audio) < min_seconds * SAMPLE_RATE:
        return False
    return float(np.sqrt(np.mean(np.square(audio)))) >= min_rms


def clean_transcript(text: str) -> str:
    text = NON_LATIN_RE.sub("", text).strip()
    if text.lower().strip(" .!?") in {h.strip(" .!?") for h in HALLUCINATIONS}:
        return ""
    return re.sub(r"\s+", " ", text)


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """Any audio format (webm, m4a, wav...) -> float32 mono 16 kHz, using ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


class STT:
    def __init__(self, model_repo: str, menu: Menu | None = None, language: str = "en",
                 use_menu_prompt: bool = True):
        import mlx_whisper  # Mac only

        self._mlx_whisper = mlx_whisper
        self.model_repo = model_repo
        self.language = language
        self.prompt = menu_prompt(menu) if (menu and use_menu_prompt) else None

    def transcribe(self, audio: np.ndarray | str) -> tuple[str, float]:
        """Returns (text, milliseconds)."""
        start = time.perf_counter()
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            language=self.language,
            initial_prompt=self.prompt,
            condition_on_previous_text=False,
            fp16=True,
        )
        ms = (time.perf_counter() - start) * 1000
        return clean_transcript(result["text"]), ms

    def warm_up(self) -> None:
        """First call downloads and loads the model; do it before the customer speaks."""
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))


def build_stt(menu: Menu) -> STT:
    from agent.config import get_settings

    cfg = get_settings()["stt"]
    return STT(cfg["model"], menu, cfg["language"], cfg["use_menu_prompt"])
