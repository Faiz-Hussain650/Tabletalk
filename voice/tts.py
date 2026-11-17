"""Text-to-speech with Kokoro (82M parameters, faster than real time on a Mac).

Download the model files once with `make models`:
    models/kokoro-v1.0.onnx and models/voices-v1.0.bin
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Iterator

import numpy as np

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PRICE_RE = re.compile(r"€\s?(\d+)(?:[.,](\d{2}))?")


def speakable(text: str) -> str:
    """Make text sound natural when spoken: '€27.50' -> '27 euros 50'."""

    def _price(m: re.Match) -> str:
        euros, cents = m.group(1), m.group(2)
        if not cents or cents == "00":
            return f"{euros} euros"
        return f"{euros} euros {int(cents)}"

    text = PRICE_RE.sub(_price, text)
    return text.replace("&", " and ").replace("*", "").strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


class TTS:
    def __init__(self, model_path: str, voices_path: str, voice: str = "af_heart", speed: float = 1.0,
                 lang: str = "en-us"):
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed
        self.lang = lang

    def synth(self, text: str) -> tuple[np.ndarray, int, float]:
        """Returns (samples, sample_rate, milliseconds)."""
        start = time.perf_counter()
        samples, sr = self.kokoro.create(speakable(text), voice=self.voice, speed=self.speed, lang=self.lang)
        return samples, sr, (time.perf_counter() - start) * 1000

    def stream_sentences(self, text: str) -> Iterator[tuple[np.ndarray, int, float]]:
        """Synthesize sentence by sentence so playback can start after the first one."""
        for sentence in split_sentences(text):
            yield self.synth(sentence)

    def warm_up(self) -> None:
        self.synth("Hello.")


def build_tts() -> TTS:
    from agent.config import ROOT, get_settings

    cfg = get_settings()["tts"]
    model_path, voices_path = ROOT / cfg["model_path"], ROOT / cfg["voices_path"]
    if not model_path.exists() or not voices_path.exists():
        raise FileNotFoundError("Kokoro model files are missing. Run `make models` first.")
    return TTS(str(model_path), str(voices_path), cfg["voice"], cfg["speed"], cfg["lang"])
