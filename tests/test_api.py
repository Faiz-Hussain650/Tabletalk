"""API smoke tests with the agent's LLM replaced by a scripted fake."""

from fastapi.testclient import TestClient

import api.main as main
from agent.config import path
from agent.llm import LLMReply, ScriptedLLM, ToolCall
from agent.order_state import Menu
from agent.text_agent import OrderingAgent


def fake_build_agent(*args, **kwargs):
    llm = ScriptedLLM([
        LLMReply("", [ToolCall("add_item", {"item_id": "st_fries", "quantity": 2})]),
        LLMReply("Two fries, €9.00. Anything else?"),
    ])
    return OrderingAgent(llm, Menu.load(path("menu")))


def test_text_chat_flow(monkeypatch):
    monkeypatch.setattr(main, "build_agent", fake_build_agent)
    client = TestClient(main.app)
    assert client.get("/health").json()["ok"]
    s = client.post("/api/session").json()
    assert "Nonna" in s["greeting"]
    r = client.post("/api/chat", json={"session_id": s["session_id"], "text": "two fries"}).json()
    assert r["cart"]["total"] == 9.0
    assert r["tool_calls"][0]["name"] == "add_item"
    assert client.post("/api/chat", json={"session_id": "nope", "text": "hi"}).status_code == 404


def test_pages_served():
    client = TestClient(main.app)
    assert "TableTalk" in client.get("/").text
    assert "Kitchen" in client.get("/kitchen").text
