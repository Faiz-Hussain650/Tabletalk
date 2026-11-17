"""Compare base vs fine-tuned (and optionally the big Ollama model) on the test split.

    uv run python -m finetune.compare --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters/v1
    uv run python -m finetune.compare --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters/v1 --ollama qwen3:8b

Every configuration is logged as an MLflow run: open `make mlflow` to see the table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.config import ROOT, get_settings
from evals.scorers import actions_exact_match, mean, parse_json_safely, percentile

TEST = ROOT / "data" / "finetune" / "test.jsonl"


def load_test(limit: int) -> list[dict]:
    rows = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def score(preds: list[str], gold: list[str], times: list[float]) -> dict[str, float]:
    parsed = [parse_json_safely(p) for p in preds]
    golds = [json.loads(g) for g in gold]
    return {
        "exact_match": mean([float(actions_exact_match(p, g)) for p, g in zip(parsed, golds, strict=True)]),
        "valid_json": mean([float(isinstance(p, list)) for p in parsed]),
        "latency_ms_p50": percentile(times, 50),
        "latency_ms_p95": percentile(times, 95),
        "n": float(len(preds)),
    }


def run_mlx(model: str, adapter: str | None, rows: list[dict]) -> tuple[list[str], list[float]]:
    from mlx_lm import generate, load

    m, tok = load(model, adapter_path=adapter) if adapter else load(model)
    preds, times = [], []
    for i, row in enumerate(rows, 1):
        prompt = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False)
        t = time.perf_counter()
        out = generate(m, tok, prompt=prompt, max_tokens=200)
        times.append((time.perf_counter() - t) * 1000)
        preds.append(out)
        if i % 25 == 0:
            print(f"   {i}/{len(rows)}")
    return preds, times


def run_ollama(model: str, rows: list[dict]) -> tuple[list[str], list[float]]:
    import ollama

    preds, times = [], []
    for row in rows:
        t = time.perf_counter()
        kwargs = {"model": model, "messages": row["messages"][:2], "options": {"temperature": 0}}
        try:
            r = ollama.chat(think=False, **kwargs)
        except TypeError:
            r = ollama.chat(**kwargs)
        times.append((time.perf_counter() - t) * 1000)
        preds.append(r.message.content)
    return preds, times


def log_run(name: str, params: dict, metrics: dict, preds: list[str], rows: list[dict]) -> None:
    out = ROOT / "results" / f"finetune_{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for p, r in zip(preds, rows, strict=True):
            f.write(json.dumps({"input": r["messages"][1]["content"][-200:], "gold": r["messages"][2]["content"],
                                "pred": p}) + "\n")
    try:
        import mlflow
    except ImportError:
        return
    cfg = get_settings()["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("kind", "finetune_compare")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out))
        run_json = Path(params.get("adapter") or "") / "run.json"
        if params.get("adapter") and run_json.exists():
            mlflow.log_params({f"train_{k}": v for k, v in json.loads(run_json.read_text()).items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter folder from train_lora.sh")
    ap.add_argument("--ollama", help="also score an Ollama model, e.g. qwen3:8b")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    rows = load_test(args.limit)
    gold = [r["messages"][2]["content"] for r in rows]
    table = []

    configs: list[tuple[str, dict]] = [("base-small", {"model": args.model, "adapter": None})]
    if args.adapter:
        configs.append((f"finetuned-{Path(args.adapter).name}", {"model": args.model, "adapter": args.adapter}))
    for name, params in configs:
        print(f"Scoring {name}...")
        preds, times = run_mlx(params["model"], params["adapter"], rows)
        metrics = score(preds, gold, times)
        try:
            log_run(name, params, metrics, preds, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] not logged: {e}")
        table.append((name, metrics))
    if args.ollama:
        print(f"Scoring ollama:{args.ollama}...")
        preds, times = run_ollama(args.ollama, rows)
        metrics = score(preds, gold, times)
        try:
            log_run(f"ollama-{args.ollama}", {"model": args.ollama, "adapter": None}, metrics, preds, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] not logged: {e}")
        table.append((f"ollama {args.ollama}", metrics))

    print("\n| Run | Exact match | Valid JSON | p50 ms | p95 ms |\n| --- | --- | --- | --- | --- |")
    for name, m in table:
        print(f"| {name} | {m['exact_match']:.3f} | {m['valid_json']:.3f} | {m['latency_ms_p50']:.0f} | {m['latency_ms_p95']:.0f} |")


if __name__ == "__main__":
    main()
