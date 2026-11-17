# TableTalk

A voice assistant that takes restaurant orders, the way a phone ordering line would. You talk to it, it asks follow-up questions, reads the order back and sends it to a kitchen screen.

Everything runs locally on my MacBook Air (M-series, 16 GB) using open-source models. No paid APIs.

I work in restaurant tech support, so I've seen what goes wrong at rush hour: phones ringing, nobody free to answer, orders getting missed. I wanted to see how far a small local model could go on this problem, and how to actually measure whether it works.

```
You:   Hi, can I get two large margherita pizzas and a coke?
Agent: Two large Margheritas and a Coke. Pickup or delivery?
You:   Actually make the coke a sprite. Pickup, my name is Marco.
Agent: So that's 2 large Margherita and 1 medium Sprite, €28.00 in total. Shall I place the order?
```

Demo video: _coming soon_

## Results

Measured with my own eval suite (25 simulated customers + 12 red-team attacks), model `qwen3:4b-instruct` on Ollama.

| Metric | Result |
| --- | --- |
| Orders exactly right | 100% (25/25) |
| Item F1 | 1.00 |
| Orders completed | 100% |
| Red-team attacks blocked | 12/12 |
| Guardrail problems that reached the customer | 0 |
| Average turns per order | 2.3 |
| Average time per turn (`make bench`) | 4.4 s (was 31.5 s) |

### How it got there

| Version | Exact match | What changed |
| --- | --- | --- |
| v1 | 0% | First run. Turned out the simulated customer (1.7B model) was copying the agent's greeting. |
| v2 | 84% | Better customer simulator, fixed a bug in my eval loop. |
| v3 | 84% | Item F1 went to 1.00 after fixing option matching ("with chicken" -> `add_chicken`). |
| v4 | 96% | Total always in the read-back, remove items by name, catch false "order placed" replies. |
| v5 | 100% | Final read-back is generated from the cart instead of by the model. |

Results vary a bit between runs (I've seen 88-100% on the same code), so I'd treat these as a range rather than a fixed number.

### Speed

The first version took ~31 s per turn. What fixed it:

- The `qwen3:4b` tag on Ollama is a thinking model. It was writing its reasoning into the reply and never calling tools. Switching to `qwen3:4b-instruct` fixed that.
- The system prompt used to include the cart, so it changed every turn and Ollama re-read ~2k tokens each time. Moving the cart into the user message keeps the prompt cacheable.
- Letting `add_item` take dish names (and several items at once) cut most orders from 5+ model calls to 2.

## How it works

```
Mic -> Silero VAD -> Whisper (mlx) -> Agent loop (Ollama) -> Guardrails -> Kokoro TTS -> Speaker
                                         |      ^
                                    tool calls  results
                                         v      |
                                  MCP menu server -> SQLite -> Kitchen display
```

Some decisions I made:

- **The cart lives in code, not in the model.** The model can only change the order through tools. Prices and totals are always calculated in `Cart`, so it can't make up a price.
- **Tools are also exposed over MCP**, so other MCP clients (or the MCP Inspector) can use the same ordering tools. The agent calls them in-process by default because it's faster.
- **Some rules are enforced in code, not just in the prompt.** The order can't be placed until the customer has heard the total and said yes. Replies with unknown prices, discount promises or allergy "it's safe" claims get rejected and regenerated.
- **Text first.** The text agent is tested and evaluated on its own; voice is a thin layer on top. Much easier to debug.
- **Fine-tuning experiment.** A 1.5B model fine-tuned with LoRA (MLX) to turn sentences into cart actions, compared against the base model on accuracy and speed.

## Running it

Needs an Apple Silicon Mac with 16 GB RAM.

```bash
brew install uv ffmpeg portaudio
brew install --cask ollama        # open the app once

make setup     # python deps, ollama model, voice model files
make test      # unit tests, no model needed
make chat      # order by typing
make api       # web app on http://localhost:8000 (+ /kitchen)
```

Other commands:

| Command | What it does |
| --- | --- |
| `make bench` | One fixed order with timings for each turn |
| `make eval` / `make eval-quick` | Simulated customers + red team |
| `make show-fails` | Prints transcripts of failed conversations |
| `make voice-ptt` / `make voice` | Voice ordering in the terminal (push-to-talk / hands-free) |
| `make record-clips` / `make eval-voice` | Record test clips and measure Whisper accuracy |
| `make inspector` | Try the MCP tools in the browser |
| `make dataset` / `make finetune` / `make compare` | Fine-tuning experiment |
| `make mlflow` | MLflow UI on port 5001 |

Models can be changed in `config/settings.yaml`, or per command: `make bench MODEL=qwen3:4b-instruct`.

Note: the model needs tool calling and should not be a thinking model. Check with `ollama show <model>`.

## Project layout

```
agent/        cart + menu, tools, agent loop, guardrails, Ollama client, MCP client, prompts
mcp_server/   the ordering tools as an MCP server
voice/        speech-to-text (mlx-whisper), text-to-speech (Kokoro), voice loop
api/          FastAPI app, ordering page and kitchen display
evals/        text + voice evals, scoring, thresholds, speed benchmark
finetune/     dataset generation, LoRA training, comparison
data/         menu, eval cases, voice scripts
tests/        unit tests
```

## Things I'd do next

- Move the voice loop to Pipecat for proper streaming
- Use the fine-tuned model inside the agent for faster turns
- Italian ordering
- An owner dashboard (popular items, failed calls, call length)

Nonna's Kitchen and the menu are made up. This is a personal learning project.
