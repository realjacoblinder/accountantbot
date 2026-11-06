"""EmbeddingGemma formatting and chunking using llama.cpp's actual tokenizer.

Limits include the embedding instruction, repeated context, and special tokens.
This module needs only the standard library; the tokenizer is served by the same
llama.cpp model used for embedding, so no second tokenizer download is needed.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

CONFIG_PATH = Path(__file__).with_name("llm.conf")
DEFAULTS = {
    "model": "embeddinggemma-2-F16",
    "ingestion_model": "embeddinggemma-2-gpu",
    "api_base": "http://127.0.0.1:8080/v1",
    "api_key": "llama.cpp",
    "timeout": 600,
    "batch_size": 32,
    "max_input_tokens": 2048,
    "tax_code_chunk_tokens": 512,
    "tax_code_overlap_tokens": 64,
    "general_chunk_tokens": 384,
    "general_overlap_tokens": 48,
    "form_chunk_tokens": 1024,
    "form_overlap_tokens": 64,
}


def load_embedding_config(config_path=None, *, purpose="query"):
    if purpose not in {"query", "ingestion"}:
        raise ValueError("Embedding purpose must be 'query' or 'ingestion'")
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    settings = DEFAULTS.copy()
    if path.exists():
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
        llm = config.get("llm", {})
        settings.update({key: llm[key] for key in ("api_base", "api_key", "timeout") if key in llm})
        settings.update(config.get("embeddings", {}))
    if purpose == "ingestion":
        settings["model"] = settings["ingestion_model"]
    settings["api_base"] = settings["api_base"].rstrip("/")
    if not settings["api_base"].endswith("/v1"):
        settings["api_base"] += "/v1"
    for key in ("batch_size", "max_input_tokens"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"embeddings.{key} must be a positive integer")
    for kind in ("tax_code", "general", "form"):
        limit, overlap = settings[f"{kind}_chunk_tokens"], settings[f"{kind}_overlap_tokens"]
        if type(limit) is not int or not 1 <= limit <= settings["max_input_tokens"]:
            raise ValueError(f"Invalid embeddings.{kind}_chunk_tokens")
        if type(overlap) is not int or not 0 <= overlap < limit:
            raise ValueError(f"Invalid embeddings.{kind}_overlap_tokens")
    return settings


def clean_document(text):
    return text.removeprefix("search_document: ").strip()


def format_document(text):
    # Source/section/identity information stays in the passage itself.
    return "title: none | text: " + clean_document(text)


def format_query(text):
    return "task: search result | query: " + text.strip()


class LlamaTokenizer:
    def __init__(self, api_base, model, api_key="llama.cpp", timeout=600):
        self.url = api_base.removesuffix("/v1").rstrip("/") + "/tokenize"
        self.model, self.api_key, self.timeout = model, api_key, timeout

    @lru_cache(maxsize=4096)
    def count(self, text):
        payload = {"model": self.model, "content": text, "add_special": True, "parse_special": True}
        request = Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                tokens = json.load(response)["tokens"]
            if not isinstance(tokens, list):
                raise ValueError("invalid tokenizer response")
            return len(tokens)
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(
                f"Cannot tokenize with {self.model} at {self.url}. "
                "Ensure the embedding server exposes /tokenize; character estimates are not used."
            ) from error


@lru_cache(maxsize=8)
def _tokenizer(api_base, model, api_key, timeout):
    return LlamaTokenizer(api_base, model, api_key, timeout)


def get_tokenizer(settings=None):
    settings = settings or load_embedding_config(purpose="ingestion")
    return _tokenizer(*(settings[key] for key in ("api_base", "model", "api_key", "timeout")))


def document_tokens(text, tokenizer=None):
    return (tokenizer or get_tokenizer()).count(format_document(text))


def _fit(text, fits):
    """Return a verified fitting prefix, including for a single huge word/row."""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(text[:middle]):
            low = middle
        else:
            high = middle - 1
    return low


def _table_headers(text):
    """Locate Markdown table headers and their contiguous table rows."""
    lines = text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    headers = []
    for index in range(len(lines) - 1):
        cells = lines[index + 1].strip().strip("|").split("|")
        if "|" not in lines[index] or not cells or not all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
            continue
        end = index + 2
        while end < len(lines) and "|" in lines[end] and lines[end].strip():
            end += 1
        headers.append((offsets[index], offsets[end] if end < len(lines) else len(text),
                        lines[index] + lines[index + 1]))
    return headers


def split_text(text, *, limit, overlap, tokenizer=None, context=""):
    """Prefer paragraphs/rows/sentences, with bounded overlap and no truncation."""
    tokenizer = tokenizer or get_tokenizer()
    text = clean_document(text)
    if not 0 <= overlap < limit:
        raise ValueError("Overlap must be nonnegative and smaller than the chunk limit")
    if not text:
        return []
    count = lambda value: document_tokens(value, tokenizer)
    if count(text) <= limit:
        return [text]
    # Keep identifiers in every split form/section without consuming its budget.
    context = context.strip()
    context = context[:_fit(context, lambda value: tokenizer.count(value) <= min(128, limit // 4))]
    tables = _table_headers(text)
    chunks, start, previous_end = [], 0, 0
    while start < len(text):
        repeated = context + "\n" if context else ""
        for table_start, table_end, header in tables:
            if table_start < start < table_end:
                repeated += header
                break
        repeated = repeated[:_fit(repeated, lambda value: count(value) <= max(1, limit // 3))]
        fits = lambda value: count(repeated + value) <= limit
        remaining = text[start:]
        size = len(remaining) if fits(remaining) else _fit(remaining, fits)
        if size == 0:
            raise ValueError("Chunk budget is too small for embedding instructions and context")
        end = start + size
        if end < len(text):
            # Start a table on the next chunk if it cannot fit beside preceding prose.
            table_break = next((a for a, b, _ in tables if start < a < end < b and a - start >= size // 3), None)
            if table_break is not None:
                end = table_break
            else:
                for separator in ("\n\n", "\n", ". ", "; ", " "):
                    boundary = text.rfind(separator, start + size // 2, end)
                    if boundary >= 0:
                        end = boundary + len(separator)
                        break
        # Tokenization at a selected boundary can differ from a longer prefix.
        if not fits(text[start:end]):
            end = start + _fit(text[start:end], fits)
        if end <= previous_end:
            # An overlap must never prevent forward progress.
            start = previous_end
            continue
        chunk = (repeated + text[start:end]).strip()
        if count(chunk) > limit:
            raise ValueError("Unable to produce a chunk within its token budget")
        chunks.append(chunk)
        previous_end = end
        if end == len(text):
            break
        if overlap:
            # Retain a suffix within the overlap budget, aligned to a line/word.
            suffix_size = _fit(text[start:end][::-1], lambda value: tokenizer.count(value[::-1]) <= overlap)
            next_start = end - suffix_size
            boundary = re.search(r"\s+", text[next_start:end])
            if boundary:
                next_start += boundary.end()
            start = max(start + 1, next_start)
        else:
            start = end
    return chunks


def chunk_documents(documents, *, kind=None, settings=None, tokenizer=None):
    """Apply the same policy to native OCR, vision, Docling, and JSONL inputs."""
    settings = settings or load_embedding_config(purpose="ingestion")
    tokenizer = tokenizer or get_tokenizer(settings)
    result = []
    for document in documents:
        metadata = document.metadata.copy()
        category = kind or ("form" if metadata.get("document_type") == "tax_form" else "general")
        context = " | ".join(f"{key}: {metadata[key]}" for key in (
            "source_file", "filename", "path", "form_type", "tax_year", "person_name", "employer_name", "page"
        ) if metadata.get(key) not in (None, ""))
        if category == "form":
            # Native OCR/Docling may not have structured identity metadata.
            # Carry short identity/header lines from the form into continuations.
            lines = document.page_content.splitlines()
            identity = []
            for index, line in enumerate(lines[:40]):
                if "|" not in line and re.search(
                    r"\b(?:tax year|20\d{2}|employee|recipient|employer|payer|account|name|w-?2|1099|1098|1040|k-1)\b",
                    line, re.IGNORECASE,
                ):
                    identity.append(line.strip())
                    if line.rstrip().endswith(":") and index + 1 < len(lines):
                        identity.append(lines[index + 1].strip())
                if len(identity) >= 6:
                    break
            context += "\n" + "\n".join(identity)
        parts = split_text(document.page_content, limit=settings[f"{category}_chunk_tokens"],
                           overlap=settings[f"{category}_overlap_tokens"], tokenizer=tokenizer, context=context)
        for index, part in enumerate(parts):
            info = metadata.copy()
            if len(parts) > 1:
                info.update(chunk_part=index + 1, chunk_parts=len(parts))
                if "doc_id" in info:
                    info["doc_id"] = f"{info['doc_id']}_part{index + 1}"
            info.update(char_count=len(part), token_count=document_tokens(part, tokenizer))
            if "token_est" in info:
                info["token_est"] = info["token_count"]
            result.append(type(document)(page_content=part, metadata=info))
    return result
