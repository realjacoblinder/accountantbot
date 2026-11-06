# ingest_user.py
# Structured PDF ingestion through the local Gemma vision service.

import hashlib
import json
import argparse
import shutil
from llm_client import create_embeddings, embed_document_batches
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_chroma import Chroma
from embedding_chunks import chunk_documents, load_embedding_config, clear_tokenizer_cache
from pdf_multimodal import (attach_pdf_images, discover_pdfs, prepare_pdf_images, store_pdf_images)
from pdf_multimodal import page_text
from pdf_vision import PDFVisionExtractor, atomic_json

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR         = "./data/user"
STORE_DIR       = "./chroma_user_docs"
COLLECTION_NAME = "user_docs"
BATCH_SIZE      = load_embedding_config(purpose="ingestion")["batch_size"]
PARALLEL_REQUESTS = load_embedding_config(purpose="ingestion")["parallel_requests"]
CLEAR_ON_START  = True

MIN_NATIVE_CHARS     = 50     # native-text classification only
IMAGE_COVERAGE_RATIO = 0.4    # if images cover >40% of page area = likely scanned

# Keywords that strongly indicate a structured tax form
TAX_FORM_MARKERS = [
    "wages, tips", "box 1", "box 2", "federal income tax withheld",
    "employer identification", "employee's social security",
    "form w-2", "form 1099", "form 1040", "form 1098",
    "nonemployee compensation", "recipient's tin",
    "schedule k-1", "ssa-1099", "ordinary dividends",
    "gross proceeds", "cost or other basis"
]
# ─────────────────────────────────────────────────────────────────────────────


class ExtractionStrategy(Enum):
    NATIVE         = "native"          # native text layout
    SCANNED    = "scanned"     # scanned layout
    FORM    = "form"     # structured tax form layout
    MIXED          = "mixed"           # mixed native and scanned layout


@dataclass
class DocumentProfile:
    path: Path
    strategy: ExtractionStrategy
    is_tax_form: bool
    page_count: int
    native_page_ratio: float    # 0.0–1.0
    has_form_fields: bool
    image_heavy_pages: int
    confidence: str             # "high" / "medium" / "low"
    reason: str                 # human-readable explanation of decision


# ── Document analyzer ─────────────────────────────────────────────────────────

def analyze_pdf(pdf_path: Path) -> DocumentProfile:
    """
    Inspect PDF structure and return a DocumentProfile with the
    layout classification for chunking; every page uses vision extraction.
    """
    pdf = fitz.open(str(pdf_path))
    page_count = len(pdf)

    # ── Signal 1: Native text density ────────────────────────────────────────
    native_pages = 0
    has_form_fields = False
    image_heavy_pages = 0
    sample_text = []

    for page in pdf:
        has_form_fields = has_form_fields or any(True for _ in (page.widgets() or []))
        text = page_text(page)
        if len(text) >= MIN_NATIVE_CHARS:
            native_pages += 1
            sample_text.append(text[:500])  # collect sample for keyword scan

        # Check image coverage ratio on this page
        page_area = page.rect.width * page.rect.height
        image_area = 0
        if page_area > 0:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                for rect in rects:
                    image_area += abs(rect.width * rect.height)
        if page_area > 0 and (image_area / page_area) > IMAGE_COVERAGE_RATIO:
            image_heavy_pages += 1

    native_ratio = native_pages / page_count if page_count > 0 else 0

    # ── Signal 2: Embedded AcroForm fields ───────────────────────────────────
    # Widgets are inspected on the actual pages; pdf_trailer() is not a dict.

    # ── Signal 3: Tax form keyword detection ─────────────────────────────────
    # Classify from native text and filename; extraction itself never uses OCR.
    full_sample = " ".join(sample_text).lower()
    filename_form = any(marker in pdf_path.name.lower() for marker in ("w-2", "w2", "1099", "1098", "1040", "k-1"))

    tax_markers_found = [m for m in TAX_FORM_MARKERS if m in full_sample]
    is_tax_form = len(tax_markers_found) >= 2 or has_form_fields or filename_form

    pdf.close()

    # ── Decision logic ────────────────────────────────────────────────────────
    if is_tax_form:
        strategy   = ExtractionStrategy.FORM
        confidence = "high" if len(tax_markers_found) >= 3 or has_form_fields else "medium"
        reason     = (
            f"Tax form detected: {len(tax_markers_found)} marker(s) found "
            f"({', '.join(tax_markers_found[:3])})"
            + (" + AcroForm fields present" if has_form_fields else "")
        )

    elif native_ratio >= 0.9:
        strategy   = ExtractionStrategy.NATIVE
        confidence = "high"
        reason     = f"Native text PDF: {native_ratio:.0%} of pages have sufficient text"

    elif native_ratio <= 0.1:
        strategy   = ExtractionStrategy.SCANNED
        confidence = "high"
        reason     = f"Fully scanned PDF: only {native_ratio:.0%} native text pages"

    else:
        strategy   = ExtractionStrategy.MIXED
        confidence = "medium"
        reason     = (
            f"Mixed PDF: {native_ratio:.0%} native text pages, "
            f"{image_heavy_pages} image-heavy page(s)"
        )

    return DocumentProfile(
        path=pdf_path,
        strategy=strategy,
        is_tax_form=is_tax_form,
        page_count=page_count,
        native_page_ratio=native_ratio,
        has_form_fields=has_form_fields,
        image_heavy_pages=image_heavy_pages,
        confidence=confidence,
        reason=reason
    )


def triage_pdf(profile: DocumentProfile, extractor) -> list[dict]:
    decisions = []
    with fitz.open(str(profile.path)) as pdf:
        if pdf.needs_pass:
            raise ValueError(f"PDF requires a password: {profile.path.name}")
        for number, page in enumerate(pdf, 1):
            try:
                decision = extractor.triage_page(page)
            except Exception as error:
                raise RuntimeError(f"Triage {profile.path.name}, page {number}: {error}") from error
            decisions.append(decision)
            print(f"    → Page {number}/{len(pdf)}: {decision['action']} — {decision['reason']}")
    identity = hashlib.sha256(profile.path.read_bytes()).hexdigest()
    source = profile.path.resolve().relative_to(Path(PDF_DIR).resolve()).as_posix()
    report_id = hashlib.sha256((source + identity).encode()).hexdigest()
    atomic_json(Path(extractor.settings["cache_directory"]) / "reports" / (report_id + ".json"), {
        "source_file": source, "source_sha256": identity,
        "model": extractor.settings["model"],
        "pages": [{"page": i+1, **decision} for i, decision in enumerate(decisions)],
    })
    return decisions


# Omitted decisions retain the original full-page extraction API for callers.
def process_pdf(profile: DocumentProfile, extractor, decisions=None) -> list[Document]:
    docs = []
    with fitz.open(str(profile.path)) as pdf:
        if pdf.needs_pass:
            raise ValueError(f"PDF requires a password: {profile.path.name}")
        if decisions is not None and len(decisions) != len(pdf):
            raise ValueError("Triage page count does not match PDF")
        for page_num, page in enumerate(pdf):
            decision = decisions[page_num] if decisions is not None else None
            if decision is not None and decision["action"] == "skip":
                print(f"    → Skipped page {page_num + 1}: {decision['reason']}")
                continue
            try:
                text = (extractor.extract_selected(page, decision) if decision is not None
                        else extractor.extract_page(page))
            except Exception as error:
                raise RuntimeError(f"{profile.path.name}, page {page_num + 1}: {error}") from error
            fields = [f"{w.field_name}: {w.field_value}" for w in page.widgets() or []
                      if w.field_value not in (None, "", "Off")]
            if fields:
                text += "\n\nCanonical PDF form fields:\n" + "\n".join(fields)
            docs.append(Document(page_content=text, metadata={
                "source_file": profile.path.resolve().relative_to(Path(PDF_DIR).resolve()).as_posix(),
                "source_pdf_path": str(profile.path.resolve()), "source_type": "user",
                "document_type": "tax_form" if profile.is_tax_form else "general",
                "strategy": "vision", "page": page_num + 1, "total_pages": len(pdf),
                "extraction_method": "gemma_vision", "extraction_model": extractor.settings["model"],
                "has_tables": "|" in text, "detection_confidence": profile.confidence,
                **({"triage_action": decision["action"], "triage_reason": decision["reason"],
                    "selected_regions": json.dumps(decision["regions"])} if decision is not None else {}),
            }))
            print(f"    → Transcribed page {page_num + 1}/{len(pdf)}")
    return docs


# ── ID + batch helpers ────────────────────────────────────────────────────────

def make_doc_id(filename: str, page: int, chunk_index: int, text: str) -> str:
    return hashlib.md5(f"{filename}::p{page}::c{chunk_index}::{text[:32]}".encode()).hexdigest()


def batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest(*, fresh=False):
    pdf_files = discover_pdfs(PDF_DIR)
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    if fresh:
        clear_tokenizer_cache()
        print("Fresh run: bypassing triage/extraction caches and rebuilding the user store.")

    print(f"Found {len(pdf_files)} PDF(s) — analyzing structure...\n")

    # Analyze all PDFs first — print a summary table before any extraction
    profiles: list[DocumentProfile] = []

    for pdf_path in pdf_files:
        profile = analyze_pdf(pdf_path)
        profiles.append(profile)

        tag = "⚠ low confidence" if profile.confidence == "low" else ""
        print(
            f"  {profile.path.name}\n"
            f"    Extraction: Gemma vision; layout profile: {profile.strategy.value} ({profile.confidence} confidence) {tag}\n"
            f"    Reason:    {profile.reason}\n"
            f"    Pages:     {profile.page_count} | "
            f"Native: {profile.native_page_ratio:.0%} | "
            f"Form fields: {profile.has_form_fields}\n"
        )

    strategy_counts = {}
    for p in profiles:
        strategy_counts[p.strategy.value] = strategy_counts.get(p.strategy.value, 0) + 1
    print("Strategy summary: " + " | ".join(f"{v}x {k}" for k, v in strategy_counts.items()) + "\n")

    # Phase 1 is eager: finish every image-embedding request before Gemma loads.
    print("Phase 1/3: Embedding all PDF page images with EmbeddingGemma 2...")
    visual_pages = prepare_pdf_images(pdf_files, PDF_DIR)
    print(f"Completed image embedding for {len(visual_pages)} pages.")

    # Phase 2 never invokes the embedding model, including its tokenizer.
    print("Phase 2a/3: Triaging all PDF pages with Gemma...")
    extractor = PDFVisionExtractor(use_cache=not fresh)
    decisions_by_pdf = []
    for profile in profiles:
        print(f"  Triaging: {profile.path.name}")
        decisions_by_pdf.append(triage_pdf(profile, extractor))
    triage_counts = {}
    for decisions in decisions_by_pdf:
        for decision in decisions:
            action = decision["action"]
            triage_counts[action] = triage_counts.get(action, 0) + 1
    print("Triage summary: " + " | ".join(f"{count} {action}" for action, count in triage_counts.items()))
    print("Phase 2b/3: Extracting selected content with Gemma...")
    all_page_docs: list[Document] = []
    for profile, decisions in zip(profiles, decisions_by_pdf):
        print(f"  Extracting: {profile.path.name} [{profile.strategy.value}]")
        all_page_docs.extend(process_pdf(profile, extractor, decisions))

    print(f"\nTotal pages extracted: {len(all_page_docs)}")

    # Keep short forms intact; split long forms with identifying context.
    # General documents use 384-token passages with 48-token overlap.
    # Phase 3 needs Gemma output; return to the embedding model only once.
    print("Phase 3/3: Tokenizing and embedding all extracted text...")
    attach_pdf_images(all_page_docs, visual_pages)
    all_chunks = chunk_documents(all_page_docs)
    print(f"Token-bounded chunks: {len(all_chunks)}\n")

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["doc_id"] = make_doc_id(
            chunk.metadata["source_file"],
            chunk.metadata.get("page", 0),
            i,
            chunk.page_content
        )

    if (CLEAR_ON_START or fresh) and Path(STORE_DIR).exists():
        shutil.rmtree(STORE_DIR)
        print("Cleared previous user store\n")

    embeddings = create_embeddings(purpose="ingestion")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=STORE_DIR,
        embedding_function=embeddings
    )
    store_pdf_images(store, visual_pages)
    print(f"Indexed {len(visual_pages)} PDF page images")

    try:
        existing = store.get(include=["metadatas"])
        existing_ids = {m["doc_id"] for m in existing["metadatas"] if "doc_id" in m}
    except Exception:
        existing_ids = set()

    new_chunks = [c for c in all_chunks if c.metadata["doc_id"] not in existing_ids]
    if not new_chunks:
        print("✓ PDF page images indexed; no additional text chunks to embed.")
        return

    print(f"Embedding {len(new_chunks)} chunks in batches of {BATCH_SIZE}...\n")
    total_added = 0
    for doc_batch, vectors in embed_document_batches(
        embeddings, batch(new_chunks, BATCH_SIZE), max_workers=PARALLEL_REQUESTS
    ):
        ids = [d.metadata["doc_id"] for d in doc_batch]
        store._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[d.page_content for d in doc_batch],
            metadatas=[d.metadata for d in doc_batch],
        )
        total_added += len(doc_batch)
        print(f"  → Progress: {total_added}/{len(new_chunks)}", end="\r")

    print(f"\n\n{'─'*50}")
    print(f"✓ Ingestion complete — {total_added} chunks added to {STORE_DIR}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ingest personal PDF documents.")
    parser.add_argument("--fresh", "--no-cache", dest="fresh", action="store_true",
                        help="Bypass saved triage/extraction results and rebuild the user vector store.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ingest(fresh=parse_args().fresh)
