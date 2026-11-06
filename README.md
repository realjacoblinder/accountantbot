# AccountantBot

AccountantBot is a local document search and question-answering project. It builds vector indexes from tax reference material and user-provided documents, then uses retrieval and a locally hosted language model to answer questions with source context. The project is designed to work with llama.cpp's OpenAI-compatible API; it does not require a hosted model account.

## Project setup

1. Install Python 3 and create an environment:

   ```sh
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Configure `llm.conf` for the local chat and embedding API services. This file contains service URLs, model aliases, and processing settings. Keep credentials and private data out of shared commits.

3. Start and configure the local model services. See [llama.cpp setup](#llamacpp-setup) below. The chat and embedding models must be available under the aliases configured in `llm.conf`.

4. Prepare the source material and build indexes as described under [Ingest documents](#ingest-documents).

## llama.cpp setup

The example configuration files and a user service unit are in [`llama-examples/`](llama-examples/): `models.ini`, `cpu-models.ini`, and `llama-server.service`. They use `$HOME/model-gguf/` for model paths; edit those paths if your GGUF files are elsewhere. The examples are starting points and may need adjustment for your hardware and llama.cpp build.

### Clone and build llama.cpp

Install the build tools and any GPU backend dependencies required for your system, then clone and build llama.cpp. For a CUDA build, for example:

```sh
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

For CPU-only use, omit `-DGGML_CUDA=ON`. Follow the llama.cpp build instructions for other GPU backends. The service example expects the executable at `~/llama.cpp/build/bin/llama-server`.

Place the model GGUFs and their matching projectors under `~/model-gguf/`, or edit the paths in the INI files. The configured aliases are:

| Alias | Purpose | Configuration |
| --- | --- | --- |
| `gemma-4-26B-A4B` | Chat, agent queries, and multimodal answer context | GPU `models.ini` |
| `gemma-4-E4B` | PDF page triage and transcription | GPU `models.ini` |
| `embeddinggemma-2-gpu` | Index building, tokenizer calls, and page-image embeddings | GPU `models.ini` |
| `embeddinggemma-2-F16` | Query embeddings | CPU `cpu-models.ini` |

The two EmbeddingGemma aliases point to the same weights and projector and use mean pooling with normalized output, so query vectors are compatible with indexed vectors. The GPU alias supports image embedding and uses four parallel slots; the CPU alias is intended for a separate query endpoint. The Gemma aliases use their matching multimodal projectors. The shared GPU defaults enable Jinja templates, flash attention, quantized KV cache, and idle unloading; the CPU defaults disable GPU offload.

Models used: 
EmbeddingGemma2 - https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/main
 - Used FP16
Gemma4-26B - https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/tree/main
Gemma4-E4B - https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF
 - Used Q4

### Install the user service

The provided `llama-server.service` runs the multi-model server on `127.0.0.1:8080`, using the GPU presets in `models.ini`. Install it as a systemd **user** unit:

```sh
mkdir -p ~/.config/systemd/user
cp /path/to/AccountantBot/llama-examples/llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
systemctl --user status llama-server.service
```

Replace `/path/to/AccountantBot` with this checkout's location. To see startup logs, run `journalctl --user -u llama-server.service -f`. To stop or restart it, use `systemctl --user stop llama-server.service` or `systemctl --user restart llama-server.service`.

The unit references `%h/llama.cpp` and `%h/llama.cpp/models.ini`, so it assumes the clone and `models.ini` are in your home directory. Copy the example `models.ini` there, or edit the service unit paths. The configured CPU embedding endpoint in `llm.conf` is `http://127.0.0.1:8081/v1`; the provided unit starts only the GPU router on port 8080. To use the CPU query alias as configured, start a second llama-server service using `cpu-models.ini` on port 8081, with the CPU alias enabled, or change `query_api_base` to a compatible embedding endpoint. Keep the service loopback-only unless you have a deliberate network access and authentication setup.

The server needs llama.cpp's OpenAI-compatible chat, embedding, and tokenization APIs. PDF flows additionally need the multimodal embedding extension and image-capable Gemma chat service. Adjust the service and model settings for the llama.cpp revision and backend you build; the API aliases must match `llm.conf` (`llm.model`, `embeddings.model`, `embeddings.ingestion_model`, and `pdf_extraction.model`).

## Use the project

Start an interactive question-answering mode with one of these entry points:

```sh
# Search tax reference material and user documents with agent tools
.venv/bin/python agentic_rag_query.py

# Restrict the agent to tax reference material
.venv/bin/python agentic_rag_query_tax_only.py

# Use the retrieval-chain interface
.venv/bin/python rag_query.py
```

`agentic_rag_query.py` prints timestamped diagnostics by default: configured
endpoints and model limits, BM25 initialization counts, per-round model input
sizes and image counts, time to first streamed chunk, streaming progress,
finish reasons and reported token usage, tool start/finish timings, separate
vector/BM25 retrieval timings and result counts, image attachments, and errors.
Each query has a run ID. Waiting model/tool calls emit a heartbeat every 10
seconds. Chunk counts and text character counts are progress indicators, not
server token counts. A zero count of exposed reasoning characters does not
prove thinking is disabled; the backend may not expose that channel.

```sh
# More metadata detail, also saved to a local file
.venv/bin/python agentic_rag_query.py --debug --log-file data/agent-debug.log

# Include bounded previews of model output, tool arguments and retrieved text
.venv/bin/python agentic_rag_query.py --debug --trace-content --log-file data/agent-debug.log

# Diagnose a streaming compatibility issue with blocking calls and heartbeats
.venv/bin/python agentic_rag_query.py --no-stream --heartbeat-seconds 5
```

Default diagnostics exclude document bodies, questions, tool argument values,
and image payloads. `--trace-content` previews can contain personal tax data;
keep those logs private. Base64 image data is omitted even from previews.
Logging does not restart the server or rebuild indexes. Streamed tool arguments
are assembled before executing a tool; failed streams are not automatically
retried, avoiding duplicate requests or tool executions.

The indexes are stored locally in `chroma_tax_code/` and `chroma_user_docs/`. They are generated data and can be rebuilt from source material.

## Ingest documents

- `uslm_chunker.py` prepares the tax reference corpus.
- `ingest_background.py` indexes the prepared tax reference data.
- `ingest_user.py` indexes user PDFs, including page-image evidence where supported by the configured embedding service.
- `user_ingest_docling.py` and `user_ingest_vlm.py` are alternative PDF ingestion workflows when their optional dependencies and model capabilities are available.

Run the tax reference preparation and ingestion when the source corpus changes, and run a user-document ingestion workflow to add or refresh personal documents. Review each script's settings and input expectations before running. Ingestion can replace or clear the corresponding local index; keep backups of any index you need to preserve. Source documents and indexes are local project data and should not be shared unless intended.

For a deliberate refresh that bypasses saved ingestion results, the main ingestion scripts accept `--fresh` (also `--no-cache`):

```sh
.venv/bin/python ingest_user.py --fresh
.venv/bin/python ingest_background.py --fresh
```

## Configuration and data

`llm.conf` is the central configuration for local model endpoints and processing options. The client code has local defaults, but configure the file to match the services you are running. Chat, text embeddings, and multimodal PDF processing may require different model capabilities.

Query responses use `llm.max_tokens` (8,192) and request `llm.reasoning_effort` (`medium`), including tool-bound agent calls. The current Gemma 4 chat template advertises no distinct reasoning-effort levels, so it may ignore the requested level; the output token cap is independent. E4B extraction keeps its separate budgets and explicitly disables thinking.

Generated indexes, source documents, and extracted content are kept in local data directories and are excluded by `.gitignore`. Keep the original source files and their corresponding indexes together when moving a working dataset. Avoid committing personal tax documents, extracted transcriptions, model files, or credentials.

## Contributing and troubleshooting

Install dependencies in an isolated environment, confirm the configured model services are reachable, and ensure the expected indexes exist before starting a query script. If retrieval results appear stale after changing embedding or chunking settings, rebuild the affected index from its source data so that all vectors use compatible settings.
