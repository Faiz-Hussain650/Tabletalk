# 🍕 TableTalk — a local voice ordering agent

**Talk to it like you'd call a restaurant. It takes your order, handles changes, answers allergy questions from real data, and sends the ticket to a kitchen screen. Everything runs on a MacBook, with open-source models and no paid APIs.**

> 🎥 Demo video: _add link_ · 📝 Write-up: _add link_

```
You:   Hi, can I get two margheritas, one without basil, and a coke?
Agent: Sure! What size for the margheritas: small, medium or large?
You:   Large. Actually make the coke a sprite.
Agent: Two large margheritas, one with no basil, and a Sprite. That's €28.00. Pickup or delivery?
```

## Results

_Fill these in from `results/` and MLflow as you go._

| Metric | Target | Result |
| --- | --- | --- |
| Order exact match (simulated customers) | ≥ 90% | |
| Item F1 | ≥ 0.95 | |
| Red-team pass rate | 100% | |
| Guardrail violations reaching the customer | 0 | |
| Speech-to-text WER on menu orders | < 8% | |
| End of speech → first audio (p50) | < 1.5 s | |
| Fine-tuned 1.5B vs base, cart-action exact match | +10 pts | |
| Cost | $0 | $0 |

## How it works

```
 Mic ─► Silero VAD ─► Whisper (mlx) ─► Agent loop (Ollama LLM) ─► Guardrails ─► Kokoro TTS ─► Speaker
                                            │  ▲
                                   tool calls│  │results
                                            ▼  │
                                   MCP menu server ─► SQLite ─► Kitchen display
                     Tracing: JSONL + Langfuse (optional) · Experiments: MLflow
```

**Key design decisions**

1. **The cart lives in code, not in the model.** The LLM can only change the order through tools (`add_item`, `update_item`, …). Prices and totals are always computed by `Cart`, so the model can't invent a price.
2. **Tools are exposed over MCP.** The same `OrderTools` run in-process for speed, or through the MCP server (`--mcp`) so any MCP client (Inspector, Claude Desktop, other agents) can use them.
3. **Guardrails in code.** Every reply is checked before it's spoken: prices must exist, no free/discount promises, allergen talk requires an allergen tool call, no "it's safe" promises. Failed replies are retried once, then replaced by a safe line.
4. **Text first, then voice.** The text agent is fully tested and evaluated; voice is a thin layer around it, so bugs are easy to locate.
5. **Evals are the product.** Simulated customers, red-team attacks and recorded voice clips are scored on every change, and results are logged to MLflow.
6. **A small fine-tuned model.** A 1.5B model is LoRA-tuned on the Mac with MLX to turn messy sentences into cart actions, then compared with the 8B model on accuracy and speed.

## Quick start (Apple Silicon Mac, 16 GB)

```bash
# 1. tools
brew install uv ffmpeg portaudio
brew install --cask ollama          # then open the Ollama app once

# 2. project
cd tabletalk
make setup                          # Python deps, Ollama models, Kokoro voice files

# 3. try it
make test                           # ~40 unit tests, no models needed
make chat                           # type an order in the terminal
make voice-ptt                      # speak an order (press Enter to talk)
make api                            # web app: http://localhost:8000  +  /kitchen
```

`make help` lists every command.

## Commands

| Command | What it does |
| --- | --- |
| `make chat` / `make chat-mcp` | Text ordering in the terminal (tools in-process / via MCP), with tool calls and timings shown |
| `make voice-ptt` / `make voice` | Voice ordering on your Mac: push-to-talk / hands-free with VAD. Add `--barge-in` (headphones) to interrupt the agent |
| `make api` | Browser ordering by text or hold-to-talk, plus live kitchen display |
| `make inspector` | Click-test every MCP tool in the MCP Inspector (needs Node.js) |
| `make eval-quick` / `make eval` / `make eval-gate` | Simulated-customer + red-team evals; `eval-gate` fails below `evals/thresholds.yaml` |
| `make record-clips` / `make eval-voice` | Record test clips from `data/evals/audio/scripts.txt`, then measure WER, menu-word recall and STT latency |
| `make dataset` / `make finetune` / `make compare` | Build data → LoRA fine-tune with MLX → compare base, fine-tuned and 8B, logged to MLflow |
| `make mlflow` | MLflow UI at http://localhost:5001 |

Change models in `config/settings.yaml` or with env vars, e.g. `TABLETALK_LLM__MAIN_MODEL=qwen3:4b make chat`.

## Project structure

```
agent/          order_state.py (menu, cart, prices) · tools.py (10 tools) · text_agent.py (agent loop)
                guardrails.py · llm.py (Ollama) · mcp_client.py · db.py · tracing.py · prompts/
mcp_server/     menu_server.py — the tools over MCP
voice/          stt.py (mlx-whisper + menu prompt) · tts.py (Kokoro) · pipeline.py (mic → speaker loop)
api/            main.py (FastAPI + WebSockets) · static/index.html · static/kitchen.html
evals/          run_text_evals.py · run_voice_evals.py · scorers.py · thresholds.yaml
finetune/       make_dataset.py · train_lora.sh · compare.py
data/           menu.json · evals/ (conversations, red team, voice scripts) · finetune/
tests/          unit tests for cart, tools, guardrails, agent loop, scorers, API
```

## Evals

- **Conversations** (`data/evals/conversations.jsonl`, 25 cases): a second local LLM plays a customer with a goal and personality; the final cart is compared with the gold order (exact match, item precision/recall/F1, completion, turns).
- **Red team** (`data/evals/redteam.jsonl`, 12 cases): prompt injection, fake discounts, price overrides, allergy "safe" promises, invented items, huge quantities, other customers' data.
- **Voice**: your own recordings → word error rate, menu-term recall and latency, with and without the menu prompt.

Every run writes `results/*.md` and logs to MLflow. Add a failing case every time you find a bug.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `connection refused` on port 11434 | Open the Ollama app, or run `ollama serve` |
| Slow first reply | Normal: the model loads once. `keep_alive: 30m` keeps it in memory |
| Replies feel slow | Try `TABLETALK_LLM__MAIN_MODEL=qwen3:4b`; close other apps |
| Mic records silence | System Settings → Privacy & Security → Microphone → allow Terminal |
| The agent hears itself | Use headphones, or don't use `--barge-in` |
| Browser mic blocked | Use `http://localhost:8000`, not your IP (browsers need https for other hosts) |
| `Kokoro model files are missing` | `make models` |
| A model name doesn't exist | Pick the current equivalent from the Ollama library and update `config/settings.yaml` |

## Roadmap

- [ ] Port the voice loop to Pipecat for streaming STT/LLM/TTS
- [ ] Serve the fine-tuned cart parser (`mlx_lm.server`) and use it inside the agent for faster turns
- [ ] Italian-language ordering
- [ ] Owner dashboard: popular items, failed orders, average call length

## Note

Nonna's Kitchen and its menu are fictional. Built on my own time as a learning project. All open-source components keep their own licenses.
