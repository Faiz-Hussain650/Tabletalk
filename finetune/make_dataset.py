"""Build the fine-tuning dataset: (current cart + customer sentence) -> JSON cart actions.

    uv run python -m finetune.make_dataset --n 3000
    uv run python -m finetune.make_dataset --n 3000 --paraphrase 300   # + LLM paraphrases via Ollama

Every example's answer is checked by applying it to a real Cart, so no broken labels get in.
Output (mlx-lm "chat" format): data/finetune/{train,valid,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from agent.config import ROOT, path
from agent.order_state import Cart, CartError, Menu

OUT = ROOT / "data" / "finetune"
NUM_WORDS = {1: ["one", "a", "1"], 2: ["two", "2", "a couple of"], 3: ["three", "3"], 4: ["four", "4"]}
OPENERS = ["", "", "hi, ", "hey, ", "um, ", "ok so ", "yeah, ", "can I get ", "I'd like ", "lemme get ", "gimme ",
           "could I have ", "I'll take "]
CLOSERS = ["", "", "", " please", " thanks", " cheers", " grazie"]

SYSTEM = (
    "You convert a customer's words into cart actions for an Italian restaurant ordering system. "
    "Reply with ONLY a JSON list. Actions: "
    '{"action":"add","item_id":str,"quantity":int,"size":"small|medium|large"|null,"modifiers":[str]} '
    '{"action":"update","line":int,"quantity":int|null,"size":str|null,"item_id":str|null,"modifiers":[str]|null} '
    '{"action":"remove","line":int}. Use [] if the words do not change the cart.'
)


def menu_context(menu: Menu) -> str:
    rows = []
    for i in menu.items:
        if not i.available:
            continue
        size = "sizes:s/m/l" if i.sizes else "no sizes"
        mods = ",".join(m.id for m in i.modifiers)
        rows.append(f"{i.id}({i.name};{size}{';mods:' + mods if mods else ''})")
    return " ".join(rows)


def spoken_name(item, rng: random.Random) -> str:  # noqa: ANN001
    options = [item.name.lower()] + [a for a in item.aliases if a.isascii()]
    return rng.choice(options)


def typo(text: str, rng: random.Random, p: float = 0.08) -> str:
    """Simulate speech-to-text noise: occasionally drop or swap a letter."""
    out = []
    for word in text.split():
        if len(word) > 4 and rng.random() < p:
            i = rng.randrange(1, len(word) - 1)
            word = word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2:]
        out.append(word)
    return " ".join(out)


def cart_text(cart: Cart) -> str:
    if not cart.order.items:
        return "[]"
    return "[" + ", ".join(f"line {li.line_no}: {li.quantity}x {li.size or ''} {li.item_id}".replace("  ", " ")
                           for li in cart.order.items) + "]"


def random_cart(menu: Menu, rng: random.Random) -> Cart:
    cart = Cart(menu)
    available = [i for i in menu.items if i.available]
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        item = rng.choice(available)
        size = rng.choice(list(item.sizes)) if item.sizes else None
        cart.add_item(item.id, rng.choice([1, 1, 2]), size)
    return cart


def make_add(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]]:
    available = [i for i in menu.items if i.available]
    actions, parts = [], []
    for _ in range(rng.choice([1, 1, 1, 2])):
        item = rng.choice(available)
        qty = rng.choice([1, 1, 1, 2, 3, 4])
        size = rng.choice(list(item.sizes)) if item.sizes and rng.random() < 0.85 else None
        mods = []
        if item.modifiers and rng.random() < 0.3:
            mod = rng.choice(item.modifiers)
            mods = [mod.id]
        words = rng.choice(NUM_WORDS[qty])
        text = f"{words} {size + ' ' if size else ''}{spoken_name(item, rng)}{'s' if qty > 1 and rng.random() < 0.5 else ''}"
        if mods:
            text += f" with {item.find_modifier(mods[0]).name}"  # type: ignore[union-attr]
        parts.append(text)
        if size is None and item.sizes and item.default_size:
            size = item.default_size
        actions.append({"action": "add", "item_id": item.id, "quantity": qty, "size": size, "modifiers": mods})
    sentence = rng.choice(OPENERS) + " and ".join(parts) + rng.choice(CLOSERS)
    return sentence, actions


def make_update_qty(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]] | None:
    if not cart.order.items:
        return None
    line = rng.choice(cart.order.items)
    qty = rng.choice([q for q in (1, 2, 3, 4) if q != line.quantity])
    item = menu.get(line.item_id)
    name = spoken_name(item, rng)
    sentence = rng.choice([
        f"actually make the {name} {rng.choice(NUM_WORDS[qty])}",
        f"can you change it to {rng.choice(NUM_WORDS[qty])} {name}",
        f"make that {rng.choice(NUM_WORDS[qty])} {name} instead",
    ])
    return sentence, [{"action": "update", "line": line.line_no, "quantity": qty, "size": None, "item_id": None, "modifiers": None}]


def make_swap(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]] | None:
    if not cart.order.items:
        return None
    line = rng.choice(cart.order.items)
    old = menu.get(line.item_id)
    same_cat = [i for i in menu.items if i.category == old.category and i.id != old.id and i.available]  # type: ignore[union-attr]
    if not same_cat:
        return None
    new = rng.choice(same_cat)
    sentence = rng.choice([
        f"swap the {spoken_name(old, rng)} for a {spoken_name(new, rng)}",
        f"actually make the {spoken_name(old, rng)} a {spoken_name(new, rng)}",
        f"instead of the {spoken_name(old, rng)} can I have {spoken_name(new, rng)}",
    ])
    return sentence, [{"action": "update", "line": line.line_no, "quantity": None, "size": None, "item_id": new.id, "modifiers": None}]


def make_size(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]] | None:
    sized = [li for li in cart.order.items if li.size]
    if not sized:
        return None
    line = rng.choice(sized)
    new_size = rng.choice([s for s in ("small", "medium", "large") if s != line.size])
    name = spoken_name(menu.get(line.item_id), rng)
    sentence = rng.choice([f"make the {name} a {new_size}", f"actually {new_size} for the {name}", f"can the {name} be {new_size}"])
    return sentence, [{"action": "update", "line": line.line_no, "quantity": None, "size": new_size, "item_id": None, "modifiers": None}]


def make_remove(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]] | None:
    if not cart.order.items:
        return None
    line = rng.choice(cart.order.items)
    name = spoken_name(menu.get(line.item_id), rng)
    sentence = rng.choice([f"remove the {name}", f"take off the {name}", f"actually no {name}", f"cancel the {name} please"])
    return sentence, [{"action": "remove", "line": line.line_no}]


def make_noop(menu: Menu, cart: Cart, rng: random.Random) -> tuple[str, list[dict]]:
    return rng.choice([
        "that's all thanks", "what time do you close", "pickup please", "my name is Marco", "yes that's right",
        "does the pesto have nuts", "how much is that", "hold on a second", "sorry what?", "no that's everything",
    ]), []


GENERATORS = [(make_add, 0.45), (make_update_qty, 0.12), (make_swap, 0.12), (make_size, 0.1), (make_remove, 0.11),
              (make_noop, 0.1)]


def validate(cart: Cart, actions: list[dict]) -> bool:
    """Apply the actions to a copy of the cart; reject the example if anything fails."""
    test = Cart(cart.menu)
    test.order = cart.order.model_copy(deep=True)
    try:
        for a in actions:
            if a["action"] == "add":
                test.add_item(a["item_id"], a["quantity"], a["size"], a["modifiers"])
            elif a["action"] == "update":
                test.update_item(a["line"], a.get("quantity"), a.get("size"), a.get("modifiers"), a.get("item_id"))
            elif a["action"] == "remove":
                test.remove_item(a["line"])
        return True
    except CartError:
        return False


def to_example(menu_ctx: str, cart: Cart, sentence: str, actions: list[dict]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"MENU: {menu_ctx}\nCART: {cart_text(cart)}\nCUSTOMER: {sentence}"},
            {"role": "assistant", "content": json.dumps(actions, separators=(",", ":"))},
        ]
    }


def paraphrase(sentences: list[str], model: str) -> list[str | None]:
    """Optional: ask a local LLM to reword sentences more naturally (same meaning)."""
    import ollama

    out: list[str | None] = []
    for s in sentences:
        try:
            r = ollama.chat(model=model, messages=[{"role": "user", "content":
                f"Reword this restaurant phone order the way a real person might say it. Keep every item, number, size "
                f"and option exactly the same. Reply with only the sentence.\n\n{s}"}])
            text = r.message.content.strip().strip('"')
            out.append(text if 0 < len(text) < 200 else None)
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def build(n: int, seed: int, n_paraphrase: int, model: str) -> list[dict]:
    rng = random.Random(seed)
    menu = Menu.load(path("menu"))
    ctx = menu_context(menu)
    rows: list[tuple[Cart, str, list[dict]]] = []
    seen: set[str] = set()
    gens, weights = zip(*GENERATORS, strict=True)
    while len(rows) < n:
        cart = random_cart(menu, rng)
        gen = rng.choices(gens, weights)[0]
        made = gen(menu, cart, rng)
        if made is None:
            continue
        sentence, actions = made
        sentence = typo(sentence, rng)
        key = cart_text(cart) + sentence
        if key in seen or not validate(cart, actions):
            continue
        seen.add(key)
        rows.append((cart, sentence, actions))
    if n_paraphrase:
        idx = rng.sample(range(len(rows)), min(n_paraphrase, len(rows)))
        new = paraphrase([rows[i][1] for i in idx], model)
        for i, s in zip(idx, new, strict=True):
            if s:
                rows[i] = (rows[i][0], s, rows[i][2])
    return [to_example(ctx, c, s, a) for c, s, a in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paraphrase", type=int, default=0, help="how many examples to reword with an LLM")
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()

    examples = build(args.n, args.seed, args.paraphrase, args.model)
    random.Random(args.seed).shuffle(examples)
    n_train, n_valid = int(len(examples) * 0.8), int(len(examples) * 0.1)
    splits = {"train": examples[:n_train], "valid": examples[n_train:n_train + n_valid],
              "test": examples[n_train + n_valid:]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> data/finetune/{name}.jsonl")


if __name__ == "__main__":
    main()
