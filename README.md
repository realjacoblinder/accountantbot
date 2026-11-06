# AccountantBot

Local tax-document retrieval and question answering using Chroma, LangChain,
and llama.cpp. All active ingestion and query scripts use the shared clients
in `llm_client.py`; Ollama is no longer required.

## Setup

Use a Python environment with the repository dependencies installed:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The optional `user_ingest_docling.py` ingestion script also requires `docling`.

Configure `llm.conf`:

- Chat and vision: `gemma-4-26B-A4B` at `http://127.0.0.1:8080/v1`.
  This is the exact model alias advertised by the running llama.cpp router.
- Embeddings: queries use `embeddings.model` (`embeddinggemma-2-F16`); initial
  vector-store generation and rebuilds use `embeddings.ingestion_model`
  (`embeddinggemma-2-gpu`). Add these aliases to your llama.cpp router, or change
  `embeddings.api_base` to its separate server URL. Start the server with embedding support and
  mean pooling for the embedding model. The client sends formatted text to
  `/v1/embeddings` and uses the same model's `/tokenize` endpoint for exact counts.
- `llm.timeout` is in seconds and allows for model loading. Context capacity
  is configured on llama-server. Agent tools require its Jinja chat template;
  the optional transcription path requires the Gemma 4 multimodal projector.
  Direct PDF page embedding requires the separate EmbeddingGemma 2 projector.

API bases accept either the server root or a URL ending in `/v1`. Local servers
use a placeholder API key; set `api_key` in either config section if needed.
No OpenAI account or hosted model is used. Missing configuration falls back to
these same local chat and embedding defaults.

## Run

```sh
.venv/bin/python agentic_rag_query.py
# Tax-code-only agent:
.venv/bin/python agentic_rag_query_tax_only.py
# Retrieval chain without agent tools:
.venv/bin/python rag_query.py
```

Ingestion entry points are `ingest_background.py`, `ingest_user.py`,
`user_ingest_docling.py`, and `user_ingest_vlm.py`. Review their configuration
before running: some user ingestion scripts clear the existing user store.

The initial vector-store build uses `embeddinggemma-2-gpu` for both tokenization
and document embedding. Query entry points use `embeddinggemma-2-F16`. These
defaults also apply when `llm.conf` is absent. Configure both aliases with the
same underlying EmbeddingGemma 2 weights, tokenizer, mean pooling, normalization,
and output dimensions, with GPU offload enabled for the ingestion alias.
The aliases must produce compatible vectors for CPU queries to search the GPU-built store.
Rebuild both Chroma indexes (`chroma_tax_code` and `chroma_user_docs`) before
querying. Existing vectors from the previous model cannot be
mixed with the new embeddings, even if their dimensions match. Back up the
old stores before rebuilding; background ingestion skips existing document
IDs, so rerunning it against an old store will not replace its vectors.
The code change itself does not delete or rebuild stored data.

`ingest_user.py.bak` is a historical backup, not an active entry point.

## Passage sizing

All active ingestion paths share the token policy in `llm.conf`:

| Content | Maximum input tokens per passage | Overlap budget |
| --- | ---: | ---: |
| Tax code | 512 | 64 |
| General user documents | 384 | 48 |
| Tax forms | 1,024 | 64 |

These are ceilings, not padding targets. Short sections and complete short forms
stay intact. Counts include the document embedding instruction, repeated context,
and BOS/EOS tokens from the actual embedding model. Overlap is aligned to words
or lines and may be shorter than its budget.

Long forms split at paragraph, table-row, sentence, or word boundaries. Source,
identity/year fields where available, and Markdown table headers are repeated in
continuations. A single row or word exceeding the limit is split without dropping
its content. Every chunk retains its original metadata and has an exact
`token_count`. Historical JSONL files are also checked and re-chunked by background
ingestion, so they cannot bypass the token limits.

The shared embedding client formats documents as `title: none | text: ...` and
search queries as `task: search result | query: ...`. Legacy `search_document:`
prefixes are removed. The client rejects inputs over `max_input_tokens` rather
than silently truncating or averaging them. It retains the model's full
768-dimensional output; HTTP embedding batches remain 32 documents.

For the embedding model, configure llama-server with:

```text
--embeddings --ctx-size 2048 --batch-size 2048 --ubatch-size 2048
--parallel 1 --pooling mean --embd-normalize 2
```

The repo's `embeddings.max_input_tokens` is 2,048, matching that server input
ceiling. Server batch sizes count tokens; `embeddings.batch_size` counts documents.
Chunking requires the `embeddinggemma-2-gpu` ingestion alias with `/tokenize`
support, including when running `uslm_chunker.py`. Ingestion and chunking never
fall back to the query alias. A tokenizer failure stops ingestion; there is no
character-count fallback or separate tokenizer download.

Back up and move both existing Chroma directories out of their configured paths
before rebuilding. Rebuild even if they already use EmbeddingGemma 2, since the
retrieval instructions and chunk boundaries have changed. Then regenerate the tax
corpus and ingest into fresh stores:

```sh
.venv/bin/python uslm_chunker.py
.venv/bin/python ingest_background.py
# Choose one user-document ingestion path:
.venv/bin/python ingest_user.py
```

Alternatively use `user_ingest_docling.py` or `user_ingest_vlm.py` for user documents.
The vision path also needs the configured chat model for text extraction; the
default native path can build stores with only the multimodal embedding model
running. The Docling path also runs its own OCR/table extraction pipeline.
Do not query the old vectors with the new retrieval formatting.

## Multimodal PDFs

`ingest_user.py` is the default PDF ingestion entry point. All three active user
ingestion paths now render every PDF page as a PNG and send it directly to
`embeddinggemma-2-gpu`, alongside their token-bounded text passages. PDF bytes are
not sent as text, and image inputs do not receive text-only instruction prefixes.
Scanned or graphical pages are indexed even when no text could be extracted.
Native text and filled AcroForm values are retained; the default native path does
not require OCR or a generative model (`pdf_ingestion.ocr_enabled` defaults to
false). Enable optional OCR only if its dependencies are installed.

Load the ingestion alias with the EmbeddingGemma 2 projector, for example:

```text
--alias embeddinggemma-2-gpu
--model /home/yaakov/model-gguf/embeddinggemma-2-F16.gguf
--mmproj /home/yaakov/model-gguf/embeddinggemma-2-mmproj-F16.gguf
--embeddings --n-gpu-layers all
--ctx-size 2048 --batch-size 2048 --ubatch-size 2048
--parallel 1 --pooling mean --embd-normalize 2 --image-max-tokens 1024
```

The server must support llama.cpp's multimodal `input[].content` extension on
`/v1/embeddings` and `/v1/chat/completions/input_tokens`. The latter counts image
tokens with the embedding model's own template/projector; it does not generate
text or load the Gemma 4 chat model. Every page is counted before embedding and
must fit `embeddings.max_input_tokens`. Missing projector/support, oversized
inputs, or invalid vectors stop preparation before the configured store-clear
step. All page vectors are prepared before writing the collection.

PDF settings in `llm.conf` control 150-DPI rendering, a 2,000-pixel longest-edge
cap, the page-image directory, optional OCR, and at most four retrieved page
images per retrieval result. Each image is embedded separately into the same
768-dimensional user-document collection, so text queries can retrieve visual
pages. Text-only queries continue using `embeddinggemma-2-F16` and do not require
its projector.

Both `agentic_rag_query.py` and `rag_query.py` attach retrieved page images to the
answering Gemma 4 model, which must have its own vision projector loaded. This
lets it read figures and tables from the page rather than guess from an image
embedding. Keyword search prefers the text passages; vector search includes both
text and visual page records. Sources include filenames and page numbers.

Keep `data/user_page_images` together with the Chroma stores when backing up or
moving the project. Cached paths are stored in metadata; missing page images
cause an explicit retrieval error. PDFs in subfolders and `.PDF` filenames are
supported, with distinct IDs for equal basenames. Rebuild the user store to add
visual records to an existing text-only index. No existing user PDFs or indexes
are modified until you explicitly run ingestion.

Run the chunking regression tests with:

```sh
python3 -m unittest discover -s tests -v
```
