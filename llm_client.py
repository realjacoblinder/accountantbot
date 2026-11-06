"""Shared LangChain chat and embedding clients for local llama.cpp servers."""

import json
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from embedding_chunks import format_document, format_query, get_tokenizer, load_embedding_config

LLM_CONFIG_PATH = Path(__file__).with_name("llm.conf")
DEFAULT_CONFIG = {
    "model": "gemma-4-26B-A4B",
    "api_base": "http://127.0.0.1:8080/v1",
    "temperature": 0.1,
    "timeout": 600,
}


def _read_config(config_path=None):
    path = Path(config_path) if config_path is not None else LLM_CONFIG_PATH
    if path.exists():
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def _api_base(url):
    # Accept both the server root and an explicit OpenAI-compatible API base.
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def load_llm_config(config_path=None):
    """Read repo-local settings, retaining llama.cpp defaults if absent."""
    config = {**DEFAULT_CONFIG, **_read_config(config_path).get("llm", {})}
    config["api_base"] = _api_base(config["api_base"])
    return config


def create_chat_model(*, temperature=None):
    """Use chat completions for text, tool calls, streaming, and vision.

    Context capacity is controlled by llama-server, not an Ollama num_ctx
    request parameter. A local placeholder API key requires no OpenAI account.
    """
    config = load_llm_config()
    return ChatOpenAI(
        model=config["model"],
        base_url=config["api_base"],
        api_key=config.get("api_key", "llama.cpp"),
        temperature=config["temperature"] if temperature is None else temperature,
        timeout=config["timeout"],
        max_retries=0,
        use_responses_api=False,
        stream_usage=False,
    )


class EmbeddingGemmaEmbeddings(Embeddings):
    """Apply retrieval instructions and reject oversized inputs without truncating."""

    def __init__(self, client, settings, tokenizer):
        self.client, self.settings, self.tokenizer = client, settings, tokenizer

    def _validate(self, texts):
        for text in texts:
            count = self.tokenizer.count(text)
            if count > self.settings["max_input_tokens"]:
                raise ValueError(
                    f"Embedding input has {count} tokens; limit is {self.settings['max_input_tokens']}. "
                    "Re-chunk the source before embedding."
                )

    def embed_documents(self, texts):
        formatted = [format_document(text) for text in texts]
        self._validate(formatted)
        return self.client.embed_documents(formatted)

    def embed_query(self, text):
        formatted = format_query(text)
        self._validate([formatted])
        return self.client.embed_query(formatted)


def create_embeddings(*, purpose="query"):
    settings = load_embedding_config(purpose=purpose)
    client = OpenAIEmbeddings(
        model=settings["model"],
        base_url=settings["api_base"],
        api_key=settings["api_key"],
        # Send text strings; tokenization uses this model's /tokenize endpoint.
        check_embedding_ctx_length=False,
        chunk_size=settings["batch_size"],
        request_timeout=settings["timeout"],
        max_retries=0,
    )
    return EmbeddingGemmaEmbeddings(client, settings, get_tokenizer(settings))
