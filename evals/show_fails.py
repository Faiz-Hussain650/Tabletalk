"""Print the transcripts of failed conversations from the latest text eval.

    uv run python -m evals.show_fails
"""

from __future__ import annotations

import json

from agent.config import ROOT


def main() -> None:
    files = sorted((ROOT / "results").glob("text_eval_*.json"))
    if not files:
        raise SystemExit("No eval results yet. Run: make eval-quick")
    data = json.loads(files[-1].read_text())
    print(f"From {files[-1].name}\n")
    for conv in data["conversations"]:
        if conv["exact_match"]:
            continue
        print(f"===== {conv['id']}  (F1={conv['item_f1']:.2f}, completed={conv['completed']})")
        for msg in conv["transcript"]:
            who, text = next(iter(msg.items()))
            print(f"  {who:>8}: {text}")
        got = [f"{i['quantity']}x {i.get('size') or ''} {i['name']}".strip() for i in conv["final_order"].get("items", [])]
        print(f"  FINAL ORDER: {got}  status={conv['final_order'].get('status')}\n")
    for r in data.get("redteam", []):
        if not r["passed"]:
            print(f"===== {r['id']} ({r['attack']}): {r['problems']}")
            for x in r["replies"]:
                print(f"  customer: {x['customer']}\n     agent: {x['agent']}")


if __name__ == "__main__":
    main()
