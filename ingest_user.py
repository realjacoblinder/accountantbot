# ingest_user.py
# Intelligence-first PDF ingestion pipeline.
# Analyzes each PDF's structure before choosing an extraction strategy.

import hashlib
import io
import shutil
from llm_client import create_embeddings
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import fitz
from langchain_core.documents import Document
from langchain_chroma import Chroma
from embedding_chunks import chunk_documents, load_embedding_config
from pdf_multimodal import (attach_pdf_images, discover_pdfs, prepare_pdf_images, store_pdf_images)
from pdf_multimodal import load_pdf_config, page_text

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR         = "./data/user"
STORE_DIR       = "./chroma_user_docs"
COLLECTION_NAME = "user_docs"
BATCH_SIZE      = load_embedding_config(purpose="ingestion")["batch_size"]
CLEAR_ON_START  = True

MIN_NATIVE_CHARS     = 50     # chars/page below this = needs OCR
IMAGE_COVERAGE_RATIO = 0.4    # if images cover >40% of page area = likely scanned
DPI_FORM             = 300    # high DPI for precise form OCR
DPI_GENERAL          = 150    # lower DPI for general scanned docs

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
    NATIVE         = "native"          # pure text PDF, extract directly
    GENERAL_OCR    = "general_ocr"     # scanned doc, basic OCR
    SPATIAL_OCR    = "spatial_ocr"     # structured tax form, spatial OCR
    MIXED          = "mixed"           # some pages native, some OCR


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
    recommended extraction strategy and the reasoning behind it.
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
    # Scan native text sample; if too little native text, do a quick low-res
    # OCR on the first page only to get enough text for keyword matching
    full_sample = " ".join(sample_text).lower()

    if len(full_sample) < 100 and page_count > 0:
        # First page low-res OCR just for classification
        ocr_engine = get_ocr_engine()
        if ocr_engine:
            page_bytes = render_page(pdf[0], dpi=100)
            try:
                result, _ = ocr_engine(page_bytes)
                if result:
                    full_sample = " ".join(r[1] for r in result).lower()
            except Exception:
                pass

    tax_markers_found = [m for m in TAX_FORM_MARKERS if m in full_sample]
    is_tax_form = len(tax_markers_found) >= 2 or has_form_fields

    pdf.close()

    # ── Decision logic ────────────────────────────────────────────────────────
    if is_tax_form:
        strategy   = ExtractionStrategy.SPATIAL_OCR
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
        strategy   = ExtractionStrategy.GENERAL_OCR
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


# ── OCR helpers ───────────────────────────────────────────────────────────────

def get_ocr_engine():
    # Page images provide visual evidence even when no OCR is installed.
    if not load_pdf_config()["ocr_enabled"]:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except ImportError:
        return None


def render_page(page: fitz.Page, dpi: int) -> bytes:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def ocr_basic(page_bytes: bytes, ocr_engine) -> str:
    from PIL import Image
    img = Image.open(io.BytesIO(page_bytes))
    result, _ = ocr_engine(img)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def ocr_spatial(page_bytes: bytes, ocr_engine) -> str:
    from PIL import Image
    img = Image.open(io.BytesIO(page_bytes))
    result, _ = ocr_engine(img)
    if not result:
        return ""

    blocks = [{"text": r[1], "x": r[0][0][0], "y": r[0][0][1]} for r in result]
    blocks.sort(key=lambda b: (round(b["y"] / 15) * 15, b["x"]))

    lines, current_line, current_y = [], [], blocks[0]["y"]
    for block in blocks:
        if abs(block["y"] - current_y) <= 15:
            current_line.append(block["text"])
        else:
            if current_line:
                lines.append("  ".join(current_line))
            current_line = [block["text"]]
            current_y = block["y"]
    if current_line:
        lines.append("  ".join(current_line))
    return "\n".join(lines)


def tesseract_fallback(page_bytes: bytes) -> str:
    if not load_pdf_config()["ocr_enabled"]:
        return ""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(page_bytes)))
    except ImportError:
        return ""


# ── Per-page extraction ───────────────────────────────────────────────────────

def extract_page(
    page: fitz.Page,
    profile: DocumentProfile,
    ocr_engine,
    tables_by_page: dict
) -> tuple[str, str]:
    """Route each page to the right extraction method based on its profile."""

    if profile.strategy == ExtractionStrategy.SPATIAL_OCR:
        native = page_text(page)
        if len(native) >= MIN_NATIVE_CHARS or profile.has_form_fields:
            text, method = native, "native_form"
        else:
            page_bytes = render_page(page, DPI_FORM)
            text = ocr_spatial(page_bytes, ocr_engine) if ocr_engine else tesseract_fallback(page_bytes)
            method = "spatial_ocr"

    elif profile.strategy == ExtractionStrategy.NATIVE:
        text = page_text(page)
        method = "native"

    elif profile.strategy in {ExtractionStrategy.GENERAL_OCR, ExtractionStrategy.MIXED}:
        native_text = page_text(page)
        if len(native_text) >= MIN_NATIVE_CHARS:
            text, method = native_text, "native"
        else:
            page_bytes = render_page(page, DPI_GENERAL)
            text = ocr_basic(page_bytes, ocr_engine) if ocr_engine else tesseract_fallback(page_bytes)
            method = "ocr"
    else:
        text = page_text(page)
        method = "native"

    # Append table markdown if available
    if page.number in tables_by_page:
        table_md = tables_by_page[page.number]
        if table_md.strip() not in text:
            text = text + "\n\n" + table_md

    return text, method


def extract_tables(pdf_path: str) -> dict[int, str]:
    try:
        import pymupdf4llm
        result = {}
        for page_data in pymupdf4llm.to_markdown(pdf_path, page_chunks=True):
            page_num = page_data.get("metadata", {}).get("page", -1)
            text = page_data.get("text", "")
            if "|" in text:
                result[page_num] = text
        return result
    except Exception:
        return {}


# ── PDF processor ─────────────────────────────────────────────────────────────

def process_pdf(profile: DocumentProfile, ocr_engine) -> list[Document]:
    pdf = fitz.open(str(profile.path))
    tables_by_page = extract_tables(str(profile.path))
    method_counts: dict[str, int] = {}
    docs = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text, method = extract_page(page, profile, ocr_engine, tables_by_page)
        method_counts[method] = method_counts.get(method, 0) + 1

        if not text.strip():
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source_file":       profile.path.resolve().relative_to(Path(PDF_DIR).resolve()).as_posix(),
                "source_pdf_path": str(profile.path.resolve()),
                "source_type":       "user",
                "document_type":     "tax_form" if profile.is_tax_form else "general",
                "strategy":          profile.strategy.value,
                "page":              page_num + 1,
                "total_pages":       profile.page_count,
                "extraction_method": method,
                "has_tables":        page_num in tables_by_page,
                "detection_confidence": profile.confidence,
            }
        ))

    pdf.close()
    counts_str = " | ".join(f"{v} {k}" for k, v in method_counts.items() if v > 0)
    print(f"    → {len(docs)} pages ({counts_str})")
    return docs


# ── ID + batch helpers ────────────────────────────────────────────────────────

def make_doc_id(filename: str, page: int, chunk_index: int, text: str) -> str:
    return hashlib.md5(f"{filename}::p{page}::c{chunk_index}::{text[:32]}".encode()).hexdigest()


def batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest():
    pdf_files = discover_pdfs(PDF_DIR)
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) — analyzing structure...\n")

    # Analyze all PDFs first — print a summary table before any extraction
    ocr_engine = get_ocr_engine()
    profiles: list[DocumentProfile] = []

    for pdf_path in pdf_files:
        profile = analyze_pdf(pdf_path)
        profiles.append(profile)

        tag = "⚠ low confidence" if profile.confidence == "low" else ""
        print(
            f"  {profile.path.name}\n"
            f"    Strategy:  {profile.strategy.value} ({profile.confidence} confidence) {tag}\n"
            f"    Reason:    {profile.reason}\n"
            f"    Pages:     {profile.page_count} | "
            f"Native: {profile.native_page_ratio:.0%} | "
            f"Form fields: {profile.has_form_fields}\n"
        )

    strategy_counts = {}
    for p in profiles:
        strategy_counts[p.strategy.value] = strategy_counts.get(p.strategy.value, 0) + 1
    print("Strategy summary: " + " | ".join(f"{v}x {k}" for k, v in strategy_counts.items()) + "\n")

    # Extract text using the detected strategy per PDF
    all_page_docs: list[Document] = []
    for profile in profiles:
        print(f"  Extracting: {profile.path.name} [{profile.strategy.value}]")
        all_page_docs.extend(process_pdf(profile, ocr_engine))

    print(f"\nTotal pages extracted: {len(all_page_docs)}")

    # Keep short forms intact; split long forms with identifying context.
    # General documents use 384-token passages with 48-token overlap.
    print("Embedding PDF page images with EmbeddingGemma 2...")
    visual_pages = prepare_pdf_images(pdf_files, PDF_DIR)
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

    if CLEAR_ON_START and Path(STORE_DIR).exists():
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
    for doc_batch in batch(new_chunks, BATCH_SIZE):
        ids = [d.metadata["doc_id"] for d in doc_batch]
        store.add_documents(documents=doc_batch, ids=ids)
        total_added += len(doc_batch)
        print(f"  → Progress: {total_added}/{len(new_chunks)}", end="\r")

    print(f"\n\n{'─'*50}")
    print(f"✓ Ingestion complete — {total_added} chunks added to {STORE_DIR}")


if __name__ == "__main__":
    ingest()
