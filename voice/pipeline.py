"""Real-time voice loop on your Mac: mic -> VAD -> Whisper -> agent -> Kokoro -> speaker.

    uv run python -m voice.pipeline --ptt         # push-to-talk (start here, easiest to debug)
    uv run python -m voice.pipeline               # hands-free with Silero VAD
    uv run python -m voice.pipeline --barge-in    # let the customer interrupt (use headphones!)

We write the loop ourselves instead of using a framework so every step is visible and timed.
Once it works, porting it to Pipecat is a good stretch goal (see README).
"""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np

from agent.config import get_settings
from agent.text_agent import build_agent
from voice.stt import SAMPLE_RATE, build_stt
from voice.tts import build_tts

CHUNK = 512  # Silero VAD expects 512-sample chunks at 16 kHz (32 ms)


class Mic:
    """Streams 16 kHz mono float32 chunks from the default microphone into a queue."""

    def __init__(self) -> None:
        import sounddevice as sd

        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=CHUNK, callback=self._callback)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self.q.put(indata[:, 0].copy())

    def __enter__(self) -> Mic:
        self.stream.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.stream.stop()
        self.stream.close()

    def clear(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()


class VAD:
    def __init__(self, threshold: float, min_silence_ms: int):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        self.torch = torch
        self.model = load_silero_vad()
        self.it = VADIterator(self.model, threshold=threshold, sampling_rate=SAMPLE_RATE,
                              min_silence_duration_ms=min_silence_ms)

    def step(self, chunk: np.ndarray) -> dict | None:
        """Returns {'start': ...}, {'end': ...} or None."""
        return self.it(self.torch.from_numpy(chunk), return_seconds=False)

    def reset(self) -> None:
        self.it.reset_states()


def listen_vad(mic: Mic, vad: VAD, preroll_chunks: int = 8, max_seconds: float = 20.0) -> np.ndarray:
    """Block until one full utterance is heard. Keeps a little audio from before speech started."""
    preroll: list[np.ndarray] = []
    speech: list[np.ndarray] = []
    speaking = False
    start = time.time()
    while True:
        chunk = mic.q.get()
        event = vad.step(chunk)
        if not speaking:
            preroll = (preroll + [chunk])[-preroll_chunks:]
            if event and "start" in event:
                speaking, speech = True, list(preroll)
                start = time.time()
        else:
            speech.append(chunk)
            if (event and "end" in event) or time.time() - start > max_seconds:
                vad.reset()
                return np.concatenate(speech)


def listen_ptt(mic: Mic) -> np.ndarray:
    input("\n[press Enter and speak]")
    mic.clear()
    print("   recording... press Enter to stop")
    chunks: list[np.ndarray] = []
    import threading

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    while not stop.is_set():
        try:
            chunks.append(mic.q.get(timeout=0.1))
        except queue.Empty:
            pass
    return np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 4, dtype=np.float32)


def speak(tts, text: str, mic: Mic | None = None, vad: VAD | None = None) -> tuple[float, bool]:  # noqa: ANN001
    """Play the reply sentence by sentence. Returns (ms to first audio, interrupted?)."""
    import sounddevice as sd

    t0 = time.perf_counter()
    first_audio_ms = 0.0
    for i, (samples, sr, _ms) in enumerate(tts.stream_sentences(text)):
        if i == 0:
            first_audio_ms = (time.perf_counter() - t0) * 1000
        sd.play(samples, sr)
        if mic is not None and vad is not None:  # barge-in: stop talking if the customer speaks
            mic.clear()
            while sd.get_stream().active:
                try:
                    event = vad.step(mic.q.get(timeout=0.05))
                except queue.Empty:
                    continue
                if event and "start" in event:
                    sd.stop()
                    vad.reset()
                    return first_audio_ms, True
        else:
            sd.wait()
    return first_audio_ms, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt", action="store_true", help="push-to-talk instead of hands-free")
    parser.add_argument("--barge-in", action="store_true", help="allow interrupting the agent (use headphones)")
    parser.add_argument("--model", help="override the Ollama model")
    args = parser.parse_args()

    cfg = get_settings()
    print("Loading models (first run downloads Whisper)...")
    agent = build_agent(model=args.model)
    stt = build_stt(agent.menu)
    tts = build_tts()
    stt.warm_up()
    tts.warm_up()
    agent.prime()
    vad = None if args.ptt else VAD(cfg["vad"]["threshold"], cfg["vad"]["min_silence_ms"])

    with Mic() as mic:
        greeting = agent.greet()
        print(f"Agent: {greeting}")
        speak(tts, greeting)
        while not agent.done:
            mic.clear()
            audio = listen_ptt(mic) if args.ptt else listen_vad(mic, vad)  # type: ignore[arg-type]
            t_end_of_speech = time.perf_counter()
            text, stt_ms = stt.transcribe(audio)
            if not text:
                continue
            print(f"You:   {text}")
            turn = agent.respond(text)
            print(f"Agent: {turn.reply}")
            first_ms, interrupted = speak(tts, turn.reply, mic if args.barge_in else None,
                                          vad if args.barge_in else None)
            total = (time.perf_counter() - t_end_of_speech) * 1000
            to_first_audio = stt_ms + turn.total_ms + first_ms
            print(f"   [stt {stt_ms:.0f}ms | agent {turn.total_ms:.0f}ms | tts-first {first_ms:.0f}ms "
                  f"| speech-end->first-audio {to_first_audio:.0f}ms | turn {total:.0f}ms"
                  f"{' | interrupted' if interrupted else ''}]")
            if agent.tracer:
                agent.tracer.log("voice_turn", input=text, output=turn.reply, stt_ms=stt_ms,
                                 agent_ms=turn.total_ms, tts_first_ms=first_ms,
                                 first_audio_ms=to_first_audio, interrupted=interrupted)
    print("\nOrder complete. Check the kitchen display at http://localhost:8000/kitchen")


if __name__ == "__main__":
    main()
