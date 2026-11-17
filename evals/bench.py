"""Speed check: runs one fixed order against your real Ollama model and prints timings per turn.

    uv run python -m evals.bench
    uv run python -m evals.bench --model qwen3:4b
    uv run python -m evals.bench --prompt v1      # compare with the old, slower prompt
"""

from __future__ import annotations

import argparse
import time

from agent.config import get_settings, path
from agent.llm import OllamaLLM
from agent.order_state import Menu
from agent.text_agent import OrderingAgent
from agent.tools import LocalToolBackend, OrderTools

SCRIPT = [
    "Hi, can I get two large margherita pizzas and a coke please?",
    "Actually make the coke a sprite.",
    "That's all.",
    "Pickup, my name is Marco.",
    "Yes, place the order.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompt", help="prompt version, e.g. v1 or v2")
    args = ap.parse_args()
    cfg = get_settings()["llm"]
    menu = Menu.load(path("menu"))
    llm = OllamaLLM(args.model or cfg["main_model"], cfg["host"], cfg["temperature"], cfg["keep_alive"],
                    num_ctx=cfg.get("num_ctx", 8192))
    print(f"Loading {llm.model}...", flush=True)
    t = time.perf_counter()
    llm.warm_up()
    print(f"model {llm.model} loaded in {time.perf_counter() - t:.1f}s\n", flush=True)

    def show(reply) -> None:  # noqa: ANN001 - live progress for every model call
        what = ", ".join(c.name for c in reply.tool_calls) or f"reply: {reply.content[:60]!r}"
        print(f"   ... model call {reply.latency_ms / 1000:.1f}s, {reply.output_tokens} tokens -> {what}", flush=True)

    llm.on_call = show
    agent = OrderingAgent(llm, menu, backend=LocalToolBackend(OrderTools(menu)), prompt_version=args.prompt)
    t = time.perf_counter()
    agent.prime()
    print(f"prompt + tools cached in {time.perf_counter() - t:.1f}s\n", flush=True)
    agent.greet()
    total = 0.0
    for line in SCRIPT:
        print(f"You:   {line}", flush=True)
        turn = agent.respond(line)
        total += turn.total_ms
        tools = ", ".join(c["name"] for c in turn.tool_calls) or "-"
        print(f"Agent: {turn.reply}\n       {turn.total_ms / 1000:.1f}s | {turn.llm_calls} LLM calls | tools: {tools}\n")
    if any("too_long_or_reasoning" in t.violations_caught for t in agent.turns):
        print("WARNING: this model 'thinks out loud' instead of calling tools. Use a non-thinking (instruct) model.")
    items = [f"{i['quantity']}x {i.get('size') or ''} {i['name']}".strip() for i in agent.cart_snapshot()["items"]]
    print(f"Average per turn: {total / len(SCRIPT) / 1000:.1f}s   Final order: {items}")


if __name__ == "__main__":
    main()
