"""Voice evals: how well does Whisper hear orders, and how fast?

1) Record test clips (reads lines from data/evals/audio/scripts.txt, 6 s each):
       uv run python -m evals.run_voice_evals --record
   Ask friends to record too: --record --speaker anna
2) Score them:
       uv run python -m evals.run_voice_evals
       uv run python -m evals.run_voice_evals --no-menu-prompt     # compare without the menu prompt
       uv run python -m evals.run_voice_evals --stt-model mlx-community/whisper-large-v3-turbo
"""

from __future__ import annotations

import argparse
import json
import time

from agent.config import ROOT, get_settings, path
from agent.order_state import Menu
from evals.scorers import mean, menu_term_recall, normalize_text, percentile

AUDIO_DIR = ROOT / "data" / "evals" / "audio"
MANIFEST = AUDIO_DIR / "manifest.jsonl"


def record(speaker: str, seconds: float) -> None:
    import sounddevice as sd
    import soundfile as sf

    scripts = [s.strip() for s in (AUDIO_DIR / "scripts.txt").read_text().splitlines() if s.strip()]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in open(MANIFEST)) if MANIFEST.exists() else 0
    print(f"Recording {len(scripts)} clips, {seconds:.0f}s each. Ctrl+C to stop at any time.\n")
    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for i, line in enumerate(scripts, 1):
            input(f"[{i}/{len(scripts)}] Press Enter, then say:\n    \"{line}\"\n")
            audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            name = f"{speaker}_{existing + i:03d}.wav"
            sf.write(AUDIO_DIR / name, audio, 16000)
            mf.write(json.dumps({"file": name, "text": line, "speaker": speaker}) + "\n")
            print(f"    saved {name}\n")


def evaluate(stt_model: str | None, use_menu_prompt: bool) -> None:
    import jiwer

    from voice.stt import STT

    cfg = get_settings()["stt"]
    menu = Menu.load(path("menu"))
    stt = STT(stt_model or cfg["model"], menu, cfg["language"], use_menu_prompt)
    stt.warm_up()
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No clips yet. Run with --record first.")
    terms = sorted({w for item in menu.items for w in [item.name, *item.aliases] if len(w) > 3}, key=len, reverse=True)

    refs, hyps, times, recalls, results = [], [], [], [], []
    for row in rows:
        text, ms = stt.transcribe(str(AUDIO_DIR / row["file"]))
        ref, hyp = normalize_text(row["text"]), normalize_text(text)
        refs.append(ref)
        hyps.append(hyp)
        times.append(ms)
        rec = menu_term_recall(ref, hyp, terms)
        if rec is not None:
            recalls.append(rec)
        results.append({**row, "hypothesis": text, "wer": jiwer.wer(ref, hyp), "ms": round(ms)})
        print(f"{row['file']}: wer={jiwer.wer(ref, hyp):.2f} {ms:.0f}ms | {text}")

    metrics = {
        "wer": jiwer.wer(refs, hyps),
        "menu_term_recall": mean(recalls),
        "stt_p50_ms": percentile(times, 50),
        "stt_p95_ms": percentile(times, 95),
        "clips": float(len(rows)),
    }
    print("\n" + "\n".join(f"{k:>18}: {v:.3f}" for k, v in metrics.items()))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = out / f"voice_eval_{stamp}.json"
    report.write_text(json.dumps({"model": stt.model_repo, "menu_prompt": use_menu_prompt, "metrics": metrics,
                                  "clips": results}, indent=2))
    try:
        import mlflow

        mcfg = get_settings()["mlflow"]
        mlflow.set_tracking_uri(mcfg["tracking_uri"])
        mlflow.set_experiment(mcfg["experiment"])
        with mlflow.start_run(run_name=f"voice-eval-{stt.model_repo.split('/')[-1]}"):
            mlflow.set_tag("kind", "voice_eval")
            mlflow.log_params({"stt_model": stt.model_repo, "menu_prompt": use_menu_prompt})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report))
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] not logged: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--speaker", default="me")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--stt-model")
    ap.add_argument("--no-menu-prompt", action="store_true")
    args = ap.parse_args()
    if args.record:
        record(args.speaker, args.seconds)
    else:
        evaluate(args.stt_model, not args.no_menu_prompt)


if __name__ == "__main__":
    main()
