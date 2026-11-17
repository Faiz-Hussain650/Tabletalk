# TableTalk - common commands. Run `make help` to list them.
.DEFAULT_GOAL := help
MODEL ?= qwen3:4b-instruct
SMALL_MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
ADAPTER ?= adapters/v1
KOKORO_URL = https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
setup:  ## install everything (Mac): Python deps + Ollama models + voice models
	uv sync
	ollama pull $(MODEL)
	$(MAKE) models
	uv run pre-commit install || true

models:  ## download Kokoro TTS model files (~340 MB) into models/
	mkdir -p models
	test -f models/kokoro-v1.0.onnx || curl -L -o models/kokoro-v1.0.onnx $(KOKORO_URL)/kokoro-v1.0.onnx
	test -f models/voices-v1.0.bin || curl -L -o models/voices-v1.0.bin $(KOKORO_URL)/voices-v1.0.bin

# ---------------------------------------------------------------- run
chat:  ## talk to the agent by typing (tools in-process)
	uv run python -m agent.text_agent --debug

chat-mcp:  ## talk to the agent by typing, tools through the MCP server
	uv run python -m agent.text_agent --mcp --debug

voice-ptt:  ## voice ordering on your Mac, push-to-talk
	uv run python -m voice.pipeline --ptt

voice:  ## voice ordering on your Mac, hands-free (VAD)
	uv run python -m voice.pipeline

api:  ## web app: http://localhost:8000 and http://localhost:8000/kitchen
	uv run uvicorn api.main:app --reload --port 8000

mcp:  ## run the MCP server on stdio
	uv run python -m mcp_server.menu_server

inspector:  ## open the MCP Inspector in your browser to click-test every tool (needs Node.js)
	npx @modelcontextprotocol/inspector uv run python -m mcp_server.menu_server

bench:  ## speed check: one fixed order, timings per turn (try MODEL=qwen3:4b)
	uv run python -m evals.bench --model $(MODEL)

# ---------------------------------------------------------------- quality
test:  ## unit tests (no Ollama needed)
	uv run pytest -q

lint:  ## ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy agent mcp_server evals

format:  ## auto-format code
	uv run ruff format .
	uv run ruff check --fix .

# ---------------------------------------------------------------- evals
eval:  ## full text evals (simulated customers + red team)
	uv run python -m evals.run_text_evals --model $(MODEL)

eval-quick:  ## 5 conversations + 5 red-team cases
	uv run python -m evals.run_text_evals --limit 5 --model $(MODEL)

eval-gate:  ## evals that fail if below thresholds (run before every release)
	uv run python -m evals.run_text_evals --model $(MODEL) --gate

record-clips:  ## record your own voice test clips
	uv run python -m evals.run_voice_evals --record

eval-voice:  ## speech-to-text accuracy and speed on your clips
	uv run python -m evals.run_voice_evals

show-fails:  ## print the conversations that failed in the latest eval
	uv run python -m evals.show_fails

# ---------------------------------------------------------------- fine-tuning
dataset:  ## build the fine-tuning dataset
	uv run python -m finetune.make_dataset --n 3000

finetune:  ## LoRA fine-tune the small model with MLX
	bash finetune/train_lora.sh $(SMALL_MODEL) 1000 $(ADAPTER)

compare:  ## base vs fine-tuned vs big model, logged to MLflow
	uv run python -m finetune.compare --model $(SMALL_MODEL) --adapter $(ADAPTER) --ollama $(MODEL)

mlflow:  ## open the MLflow UI at http://localhost:5001
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# ---------------------------------------------------------------- docker
up:  ## start MLflow server in Docker (optional)
	docker compose up -d

down:  ## stop Docker services
	docker compose down

clean:  ## delete local DB, traces and results
	rm -rf data/tabletalk.db traces/*.jsonl results/*

.PHONY: help setup models bench show-fails chat chat-mcp voice-ptt voice api mcp inspector test lint format eval eval-quick eval-gate record-clips eval-voice dataset finetune compare mlflow up down clean
