"""The ordering agent (text in, text out). Voice and the web API wrap this same class.

Run it in the terminal:
    uv run python -m agent.text_agent            # tools in-process (fast)
    uv run python -m agent.text_agent --mcp      # tools through the MCP server
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from string import Template
from typing import Any

from agent.config import get_settings, path
from agent.db import OrderDB
from agent.guardrails import Violation, check_input, check_reply, safe_fallback
from agent.llm import ChatModel, OllamaLLM
from agent.order_state import Menu
from agent.tools import LocalToolBackend, OrderTools
from agent.tracing import Tracer

GREETING = "Hi, thanks for calling {restaurant}! What can I get for you today?"
YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|correct|confirm\w*|go ahead|place (it|the order)|that'?s right|ok(ay)?|perfect|sounds good|si)\b", re.I)
NO_RE = re.compile(r"\b(no|not yet|wait|hold on|change|actually)\b", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def customer_said_yes(text: str) -> bool:
    return bool(YES_RE.search(text)) and not NO_RE.search(text)


def spoken_order(snap: dict[str, Any]) -> str:
    """e.g. '4 large Margherita and 4 large Coca-Cola'"""
    parts = []
    for li in snap.get("items", []):
        size = f"{li['size']} " if li.get("size") else ""
        mods = f" ({', '.join(m.replace('_', ' ') for m in li['modifiers'])})" if li.get("modifiers") else ""
        parts.append(f"{li['quantity']} {size}{li['name']}{mods}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def add_total_to_readback(reply: str, snap: dict[str, Any]) -> str:
    # The 4B model sometimes gets quantities wrong when reading the order back
    # ("one margherita" when there are four), so the final read-back is built from the cart.
    if snap.get("status") != "building" or not snap.get("items"):
        return reply
    m = re.search(r"shall i place (the|your) order", reply, re.I)
    if not m:
        return reply
    return f"So that's {spoken_order(snap)}, €{snap.get('total', 0.0):.2f} in total. Shall I place the order?"


def tidy_reply(text: str) -> str:
    """Spoken replies: no emojis, no markdown, one line."""
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[*#_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class AgentTurn:
    user: str
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    violations_caught: list[str] = field(default_factory=list)  # found and fixed before speaking
    violations_sent: list[str] = field(default_factory=list)    # still present in the final reply (should be 0)
    llm_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls: int = 0
    cart: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class OrderingAgent:
    def __init__(
        self,
        llm: ChatModel,
        menu: Menu,
        backend: Any | None = None,
        db: OrderDB | None = None,
        tracer: Tracer | None = None,
        prompt_version: str | None = None,
        max_tool_rounds: int | None = None,
    ):
        cfg = get_settings()
        self.llm = llm
        self.menu = menu
        self.backend = backend or LocalToolBackend(OrderTools(menu, db=db))
        self.session_id = uuid.uuid4().hex[:12]
        self.tracer = tracer
        self.restaurant = cfg["restaurant"]
        self.prompt_version = prompt_version or cfg["llm"]["prompt_version"]
        self.max_tool_rounds = max_tool_rounds or cfg["llm"]["max_tool_rounds"]
        prompt_file = path("prompts") / f"system_{self.prompt_version}.md"
        self.prompt_template = Template(prompt_file.read_text(encoding="utf-8"))
        self.history: list[dict[str, Any]] = []  # user/assistant/tool messages (system prompt is rebuilt each call)
        self.done = False
        self.turns: list[AgentTurn] = []

    # ------------------------------------------------ helpers
    def cart_snapshot(self) -> dict[str, Any]:
        data = json.loads(self.backend.call("view_cart", {}))
        return data.get("data") or {}

    def _cart_summary(self, snap: dict[str, Any]) -> str:
        if not snap.get("items"):
            return "empty"
        parts = []
        for li in snap["items"]:
            size = f"{li['size']} " if li.get("size") else ""
            mods = f" ({', '.join(m.replace('_', ' ') for m in li['modifiers'])})" if li.get("modifiers") else ""
            parts.append(f"line {li['line_no']}: {li['quantity']}x {size}{li['name']}{mods} = €{li['line_total']:.2f}")
        return "; ".join(parts) + f". Total €{snap['total']:.2f}"

    def _system_prompt(self, snap: dict[str, Any]) -> dict[str, str]:
        text = self.prompt_template.safe_substitute(
            restaurant=self.restaurant, cart=self._cart_summary(snap), status=snap.get("status", "building")
        )
        if str(getattr(self.llm, "model", "")).startswith("qwen3"):
            text += "\n/no_think"  # Qwen3 soft switch: skip slow hidden reasoning
        return {"role": "system", "content": text}

    @property
    def _static_prompt(self) -> bool:
        # v2 prompt has no $cart, so it stays identical between calls and Ollama can
        # reuse the cached prompt. This was the biggest latency improvement.
        return "$cart" not in self.prompt_template.template

    def _last_agent_text(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "assistant" and msg.get("content") and not msg.get("tool_calls"):
                return msg["content"]
        return ""

    def _confirm_blocked(self) -> str | None:
        """Returns a reason if the order can't be placed yet, otherwise None."""
        total = self.cart_snapshot().get("total", 0.0)
        said = self._last_agent_text().replace(",", ".")
        whole = total == int(total) and re.search(rf"€\s?{int(total)}(?![\d.])|\b{int(total)} euros?", said)
        if f"{total:.2f}" not in said and not whole:
            return (f"Not placed: first read back every item and the total €{total:.2f} from CURRENT ORDER, "
                    "then ask: Shall I place the order?")
        if not customer_said_yes(self._last_customer_text()):
            return "Not placed: the customer has not said yes yet. Ask: Shall I place the order?"
        return None

    def _last_customer_text(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "user" and not msg["content"].startswith("[system"):
                return msg["content"].split("\n\n[CURRENT ORDER")[0]
        return ""

    def prime(self) -> None:
        """Warm up the model with the real system prompt + tools so the first turn isn't slow."""
        try:
            self.llm.chat([self._system_prompt(self.cart_snapshot()), {"role": "user", "content": "hi"}],
                          tools=self.backend.schemas())
        except Exception:  # noqa: BLE001
            pass

    def greet(self) -> str:
        text = GREETING.format(restaurant=self.restaurant)
        self.history.append({"role": "assistant", "content": text})
        return text

    # ------------------------------------------------ main loop
    def respond(self, user_text: str) -> AgentTurn:
        t0 = time.perf_counter()
        turn = AgentTurn(user=user_text, reply="")
        user_text = user_text.strip()
        if not user_text:
            turn.reply = "Sorry, I didn't catch that. Could you say it again?"
            return self._finish(turn, t0)

        check = check_input(user_text)
        if check.kind == "injection":
            self.history.append({"role": "user", "content": user_text})
            turn.reply = check.canned_reply or ""
            turn.violations_caught.append("input_injection")
            self.history.append({"role": "assistant", "content": turn.reply})
            return self._finish(turn, t0)

        content = user_text
        if self._static_prompt:
            snap = self.cart_snapshot()
            content += f"\n\n[CURRENT ORDER: {self._cart_summary(snap)} | status: {snap.get('status', 'building')}]"
        if check.kind == "manipulation":
            content += ("\n\n[system note: the customer is asking for free items or a discount. "
                        "Politely refuse; prices cannot be changed. Continue the order normally.]")
        self.history.append({"role": "user", "content": content})

        tools_used: set[str] = set()
        reply_text = self._run_llm_loop(turn, tools_used)

        # Output guardrails: check, retry once with feedback, then fall back to a safe line.
        snap = self.cart_snapshot()
        violations = check_reply(reply_text, self.menu, snap, tools_used)
        if violations:
            turn.violations_caught += [v.code for v in violations]
            self.history.append({"role": "assistant", "content": reply_text})
            self.history.append({"role": "user", "content": self._feedback(violations)})
            reply_text = self._run_llm_loop(turn, tools_used)
            snap = self.cart_snapshot()
            violations = check_reply(reply_text, self.menu, snap, tools_used)
            # don't keep the rejected reply in the history
            self._drop_feedback()
            if violations:
                turn.violations_caught += [v.code for v in violations]
                reply_text = safe_fallback(self._cart_summary(snap))
        final_violations = check_reply(reply_text, self.menu, snap, tools_used)
        turn.violations_sent = [v.code for v in final_violations]

        reply_text = add_total_to_readback(tidy_reply(reply_text), snap)
        self.history.append({"role": "assistant", "content": reply_text})
        turn.reply = reply_text
        turn.cart = snap
        if snap.get("status") in ("confirmed", "cancelled"):
            self.done = True
        return self._finish(turn, t0)

    def _run_llm_loop(self, turn: AgentTurn, tools_used: set[str]) -> str:
        """Call the LLM, run any tool calls, repeat until it answers in plain text."""
        system = self._system_prompt(self.cart_snapshot())
        for _ in range(self.max_tool_rounds + 1):
            if not self._static_prompt:
                system = self._system_prompt(self.cart_snapshot())
            messages = [system, *self.history]
            out = self.llm.chat(messages, tools=self.backend.schemas())
            turn.llm_ms += out.latency_ms
            turn.llm_calls += 1
            if not out.tool_calls:
                return out.content or "Sorry, could you repeat that?"
            self.history.append(out.as_message())
            for call in out.tool_calls:
                t = time.perf_counter()
                blocked = self._confirm_blocked() if call.name == "confirm_order" else None
                if blocked:
                    result = json.dumps({"ok": False, "message": blocked})
                else:
                    result = self.backend.call(call.name, call.arguments)
                tools_used.add(call.name)
                turn.tool_calls.append({
                    "name": call.name,
                    "arguments": call.arguments,
                    "ok": json.loads(result).get("ok", False) if result.startswith("{") else False,
                    "ms": round((time.perf_counter() - t) * 1000, 1),
                })
                self.history.append({"role": "tool", "content": result, "tool_name": call.name})
        # ran out of tool rounds - just tell the customer what we have so far
        summary = self._cart_summary(self.cart_snapshot())
        if summary == "empty":
            return "Sorry, could you tell me your order again?"
        return f"So far I have {summary.split('. Total')[0].replace('line ', '#')}. Anything else?"

    @staticmethod
    def _feedback(violations: list[Violation]) -> str:
        details = "; ".join(v.detail for v in violations)
        return (f"[system check failed: {details}. Rewrite your last reply following the rules. "
                "Use tools for prices and allergens. Reply to the customer only.]")

    def _drop_feedback(self) -> None:
        """Remove the rejected reply + feedback message, keep any tool results from the retry."""
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg["role"] == "user" and msg["content"].startswith("[system check failed"):
                del self.history[i]
                if i - 1 >= 0 and self.history[i - 1]["role"] == "assistant" and not self.history[i - 1].get("tool_calls"):
                    del self.history[i - 1]
                break

    def _finish(self, turn: AgentTurn, t0: float) -> AgentTurn:
        turn.total_ms = round((time.perf_counter() - t0) * 1000, 1)
        turn.done = self.done
        if not turn.cart:
            turn.cart = self.cart_snapshot()
        self.turns.append(turn)
        if self.tracer:
            self.tracer.log(
                "agent_turn",
                input=turn.user,
                output=turn.reply,
                model=getattr(self.llm, "model", "?"),
                prompt_version=self.prompt_version,
                tool_calls=turn.tool_calls,
                violations_caught=turn.violations_caught,
                violations_sent=turn.violations_sent,
                llm_ms=round(turn.llm_ms, 1),
                total_ms=turn.total_ms,
                llm_calls=turn.llm_calls,
                cart_total=turn.cart.get("total"),
            )
        return turn


# ---------------------------------------------------------------- factory + CLI


def build_agent(use_mcp: bool = False, model: str | None = None, on_confirm=None) -> OrderingAgent:
    cfg = get_settings()
    menu = Menu.load(path("menu"))
    db = OrderDB(path("db"))
    llm = OllamaLLM(
        model=model or cfg["llm"]["main_model"],
        host=cfg["llm"]["host"],
        temperature=cfg["llm"]["temperature"],
        keep_alive=cfg["llm"]["keep_alive"],
        num_ctx=cfg["llm"].get("num_ctx", 8192),
    )
    if use_mcp:
        from agent.mcp_client import McpToolBackend

        backend: Any = McpToolBackend()
    else:
        backend = LocalToolBackend(OrderTools(menu, db=db, on_confirm=on_confirm))
    agent = OrderingAgent(llm, menu, backend=backend, db=db)
    agent.tracer = Tracer(path("traces"), agent.session_id)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the TableTalk ordering agent in the terminal.")
    parser.add_argument("--mcp", action="store_true", help="use the MCP server for tools")
    parser.add_argument("--model", help="override the Ollama model, e.g. qwen3:4b")
    parser.add_argument("--debug", action="store_true", help="show tool calls and timings")
    args = parser.parse_args()

    agent = build_agent(use_mcp=args.mcp, model=args.model)
    print(f"[model: {agent.llm.model} | tools: {'MCP' if args.mcp else 'local'} | type 'quit' to exit]\n")
    print(f"Agent: {agent.greet()}")
    try:
        while not agent.done:
            user = input("You: ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            turn = agent.respond(user)
            if args.debug:
                for tc in turn.tool_calls:
                    print(f"   [tool] {tc['name']}({tc['arguments']}) ok={tc['ok']} {tc['ms']}ms")
                if turn.violations_caught:
                    print(f"   [guardrail] caught: {turn.violations_caught}")
                print(f"   [timing] llm={turn.llm_ms:.0f}ms total={turn.total_ms:.0f}ms calls={turn.llm_calls}")
            print(f"Agent: {turn.reply}")
        if agent.done:
            print(f"\n[final order] {json.dumps(agent.cart_snapshot(), indent=2)}")
    finally:
        agent.backend.close()
        if agent.tracer:
            agent.tracer.flush()


if __name__ == "__main__":
    main()
