"""Turn retrieved text/page records into cited text and bounded visual evidence."""

from pathlib import Path

from pdf_multimodal import image_data_url, load_pdf_config


def keyword_documents(documents):
    text = [d for d in documents if d.metadata.get("embedding_type") != "page_image"]
    return text or documents


def source_text(document):
    metadata = document.metadata
    source = metadata.get("source_file") or metadata.get("filename") or metadata.get("path", "tax_code")
    page = f", page {metadata['page']}" if "page" in metadata else ""
    return f"[SOURCE: {source}{page}]\n{document.page_content}"


def retrieval_content(documents, *, settings=None):
    settings = settings or load_pdf_config()
    content, seen = [], set()
    image_root = Path(settings["image_directory"]).resolve()
    for document in documents:
        content.append({"type": "text", "text": source_text(document)})
        image = document.metadata.get("page_image_path")
        if not image or image in seen or len(seen) >= settings["max_retrieved_pages"]:
            continue
        path = Path(image).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError("Retrieved PDF image is outside the configured page-image directory")
        # Missing images must be repaired/re-ingested, not silently discarded.
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        seen.add(image)
    return content or [{"type": "text", "text": "No results found."}]
