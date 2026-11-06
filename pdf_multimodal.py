"""Render PDFs and index page images beside token-bounded text passages.

Images are embedded directly by EmbeddingGemma 2, not transcribed by a chat
model. Text queries can retrieve these vectors in the same embedding space.
"""

import base64
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from embedding_chunks import CONFIG_PATH, load_embedding_config

ROOT = Path(__file__).resolve().parent
PDF_DEFAULTS = {"render_dpi": 150, "max_image_edge": 2000,
                "image_directory": "data/user_page_images", "max_retrieved_pages": 4}
PDF_IMAGE_EMBEDDING_WORKERS = 2


def load_pdf_config():
    settings = PDF_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as file:
            settings.update(json.load(file).get("pdf_ingestion", {}))
    for key in ("render_dpi", "max_image_edge", "max_retrieved_pages"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"pdf_ingestion.{key} must be a positive integer")
    directory = Path(settings["image_directory"])
    settings["image_directory"] = directory if directory.is_absolute() else ROOT / directory
    return settings


def image_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def discover_pdfs(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def page_text(page):
    """Keep native PDF text and canonical filled-form values without OCR loss."""
    text = page.get_text("text").strip()
    fields = []
    for widget in page.widgets() or []:
        if widget.field_value not in (None, "", "Off"):
            fields.append(f"{widget.field_name}: {widget.field_value}")
    if fields:
        text += "\n\nPDF form fields:\n" + "\n".join(fields)
    return text.strip()


class PDFImageEmbedder:
    def __init__(self, settings=None):
        self.settings = settings or load_embedding_config(purpose="ingestion")

    def _post(self, route, payload):
        request = Request(self.settings["api_base"] + route,
                          data=json.dumps({"model": self.settings["model"], **payload}).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + self.settings["api_key"]})
        try:
            with urlopen(request, timeout=self.settings["timeout"]) as response:
                return json.load(response)
        except (HTTPError, OSError, ValueError) as error:
            raise RuntimeError(
                f"PDF image embedding failed for {self.settings['model']}. "
                "Check server connectivity, --embeddings, the EmbeddingGemma 2 --mmproj, "
                "multimodal input-token counting support, and context/physical batch limits."
            ) from error

    def embed_page(self, image_path):
        # A single page image per input; no text-only instruction prefix on images.
        content = [{"type": "image_url", "image_url": {"url": image_data_url(image_path)}}]
        # /tokenize only handles text. This route counts actual image tokens using
        # the embedding model's own template and projector, without generating text.
        count = self._post("/chat/completions/input_tokens", {
            "messages": [{"role": "user", "content": content}],
        }).get("input_tokens")
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid multimodal input-token count")
        if count > self.settings["max_input_tokens"]:
            raise ValueError(f"PDF page needs {count} tokens; embedding limit is "
                             f"{self.settings['max_input_tokens']}. Reduce --image-max-tokens "
                             "or increase context and physical batch size together.")
        response = self._post("/embeddings", {
            "input": [{"content": content}], "encoding_format": "float",
        })
        data = response.get("data", [])
        if len(data) != 1 or data[0].get("index") != 0:
            raise ValueError("Expected exactly one PDF-page embedding")
        vector = data[0].get("embedding", [])
        if len(vector) != 768 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            raise ValueError("PDF page embedding must contain 768 finite values")
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            raise ValueError("PDF page embedding is a zero vector")
        return [v / norm for v in vector], count


@dataclass
class PageVector:
    doc_id: str
    text: str
    metadata: dict
    embedding: list


def prepare_pdf_images(pdf_files, source_root, *, embedder=None, pdf_settings=None):
    """Embed pages with at most two concurrent calls before changing the store."""
    import fitz

    embedder = embedder or PDFImageEmbedder()
    settings = pdf_settings or load_pdf_config()
    source_root = Path(source_root).resolve()
    page_inputs = []
    for filename in pdf_files:
        path = Path(filename).resolve()
        source = path.relative_to(source_root).as_posix()
        with path.open("rb") as file:
            file_hash = hashlib.file_digest(file, "sha256").hexdigest()
        identity = hashlib.sha256((source + "\0" + file_hash).encode()).hexdigest()
        directory = Path(settings["image_directory"]) / identity
        directory.mkdir(parents=True, exist_ok=True)
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"PDF requires a password: {source}")
            for index, page in enumerate(pdf):
                scale = min(settings["render_dpi"] / 72,
                            settings["max_image_edge"] / max(page.rect.width, page.rect.height))
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB,
                                        alpha=False, annots=True)
                image = directory / f"page-{index + 1:04d}.png"
                pixmap.save(str(image))
                doc_id = f"pdf_image_{identity}_{index + 1}"
                text = f"PDF page {index + 1} of {source}\n" + page_text(page)
                metadata = {
                    "doc_id": doc_id, "source_file": source, "filename": source,
                    "source_pdf_path": str(path), "source_type": "user",
                    "page": index + 1, "total_pages": len(pdf),
                    "page_image_path": str(image.resolve()), "embedding_type": "page_image",
                    "embedding_model": embedder.settings["model"],
                }
                # Rendering and PyMuPDF access stay on this thread. Only network
                # embedding work runs concurrently, capped to two pages at once.
                page_inputs.append((image, doc_id, text.strip(), metadata))

    with ThreadPoolExecutor(max_workers=PDF_IMAGE_EMBEDDING_WORKERS) as executor:
        embedded = list(executor.map(lambda item: embedder.embed_page(item[0]), page_inputs))

    result = []
    for (image, doc_id, text, metadata), (vector, tokens) in zip(page_inputs, embedded):
        metadata["token_count"] = tokens
        result.append(PageVector(doc_id, text, metadata, vector))
    return result


def attach_pdf_images(documents, pages):
    lookup = {(p.metadata["source_pdf_path"], p.metadata["page"]): p for p in pages}
    for document in documents:
        key = (document.metadata.get("source_pdf_path"), document.metadata.get("page"))
        if key in lookup:
            document.metadata["page_image_path"] = lookup[key].metadata["page_image_path"]
    return documents


def store_pdf_images(store, pages):
    # Precomputed image vectors bypass the text-only LangChain embedding method.
    # One upsert per page avoids accumulating base64 images or large batches.
    for page in pages:
        store._collection.upsert(ids=[page.doc_id], embeddings=[page.embedding],
                                 documents=[page.text], metadatas=[page.metadata])
