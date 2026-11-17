"""Thin wrapper around the Ollama client: chat with tools, and JSON-schema output."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"function": {"name": t.name, "arguments": t.arguments}} for t in self.tool_calls]
        return msg


class ChatModel(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


class OllamaLLM:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2,
                 keep_alive: str = "30m", think: bool = False, num_ctx: int = 8192,
                 num_predict: int = 400, timeout_s: float = 120.0):
        import ollama  # imported here so tests can run without Ollama installed

        self.model = model
        # timeout: never hang forever. num_predict: a phone reply never needs more than ~400 tokens;
        # without a cap, a confused model can ramble for minutes.
        self.client = ollama.Client(host=host, timeout=timeout_s)
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
        self.on_call = None  # optional callback(reply) for live progress printing
        self.keep_alive = keep_alive
        self.think = think

    def _chat_raw(self, **kwargs: Any) -> Any:
        try:
            return self.client.chat(model=self.model, options=self.options, keep_alive=self.keep_alive,
                                    think=self.think, **kwargs)
        except TypeError:  # older ollama-python without `think` - upgrade it: `uv add "ollama>=0.5"`
            return self.client.chat(model=self.model, options=self.options, keep_alive=self.keep_alive, **kwargs)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        start = time.perf_counter()
        resp = self._chat_raw(messages=messages, tools=tools or None)
        latency = (time.perf_counter() - start) * 1000
        msg = resp.message
        calls = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=tc.function.name, arguments=dict(args or {})))
        content = THINK_RE.sub("", msg.content or "").strip()
        if "<think>" in content:  # unfinished thinking block (hit num_predict): drop it
            content = content.split("<think>")[0].strip()
        reply = LLMReply(
            content=content,
            tool_calls=calls,
            latency_ms=latency,
            prompt_tokens=getattr(resp, "prompt_eval_count", 0) or 0,
            output_tokens=getattr(resp, "eval_count", 0) or 0,
        )
        if self.on_call:
            self.on_call(reply)
        return reply

    def json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> Any:
        """Force output that matches a JSON schema (Ollama structured outputs)."""
        resp = self._chat_raw(messages=messages, format=schema)
        text = THINK_RE.sub("", resp.message.content or "").strip()
        return json.loads(text)

    def warm_up(self) -> None:
        """Load the model into memory so the first real turn is fast."""
        self._chat_raw(messages=[{"role": "user", "content": "hi"}])


class ScriptedLLM:
    """A fake LLM for unit tests: returns pre-written replies in order."""

    def __init__(self, replies: list[LLMReply]):
        self.model = "scripted"
        self.replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        self.calls.append(list(messages))
        if not self.replies:
            return LLMReply(content="Anything else I can get you?")
        return self.replies.pop(0)
