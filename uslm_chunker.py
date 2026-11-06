"""Prepare US Tax Code XML as passages bounded by actual embedding tokens.

Preserves section hierarchy, splitting oversized leaves at paragraph, sentence,
or word boundaries. Embedding instructions and repeated source context count
against the shared 512-token budget. Output text has no model-specific prefix;
the shared embedding client applies EmbeddingGemma's document instruction.
"""

import xml.etree.ElementTree as ET
import json
import re
import sys
from pathlib import Path
from embedding_chunks import document_tokens, get_tokenizer, load_embedding_config, split_text

# ── CONFIG ────────────────────────────────────────────────────────────────
FILE            = "usc26.xml"          # <- your input file
OUTPUT          = "./data/background/chunks.jsonl"

SETTINGS = load_embedding_config(purpose="ingestion")
TOKEN_LIMIT = SETTINGS["tax_code_chunk_tokens"]
TOKEN_OVERLAP = SETTINGS["tax_code_overlap_tokens"]

USLM_NS         = "{http://xml.house.gov/schemas/uslm/1.0}"

HIERARCHY = [
    "section",
    "subsection",
    "paragraph",
    "subparagraph",
    "clause",
    "subclause",
]

SKIP_TAGS = {
    "meta", "metadata", "toc", "tocItem",
    "sourceCredit", "notes", "note",
    "img", "signature",
}

# ── HELPERS ───────────────────────────────────────────────────────────────
def strip_ns(tag: str) -> str:
    return re.sub(r"\{[^}]+\}", "", tag)

def ns(local: str) -> str:
    return f"{USLM_NS}{local}"

def get_text(elem) -> str:
    """Collect all inner text, skipping SKIP_TAGS subtrees."""
    parts = []
    for e in elem.iter():
        if strip_ns(e.tag) in SKIP_TAGS:
            continue
        if e.text and e.text.strip():
            parts.append(e.text.strip())
        if e.tail and e.tail.strip():
            parts.append(e.tail.strip())
    return " ".join(parts)

def token_count(text: str) -> int:
    return document_tokens(text, get_tokenizer(SETTINGS))

def get_label(elem) -> str:
    num_elem = elem.find(ns("num")) or elem.find("num")
    if num_elem is not None and num_elem.text:
        return num_elem.text.strip().rstrip(".")
    ident = elem.attrib.get(f"{USLM_NS}identifier") or elem.attrib.get("identifier") or ""
    if ident:
        return ident.split("/")[-1]
    return strip_ns(elem.tag)

def get_heading(elem) -> str:
    h = elem.find(ns("heading")) or elem.find("heading")
    if h is not None and h.text:
        return h.text.strip()
    return ""

def get_identifier(elem) -> str:
    return (
        elem.attrib.get(f"{USLM_NS}identifier")
        or elem.attrib.get("identifier")
        or elem.attrib.get("id")
        or ""
    )

# ── CORE CHUNKER ──────────────────────────────────────────────────────────
_chunks  = []
_counter = [0]

def emit(path: str, identifier: str, text: str, context: str = ""):
    """Record verified passages, repeating section context on split leaves."""
    text = re.sub(r"\s{2,}", " ", text).strip()
    parts = split_text(text, limit=TOKEN_LIMIT, overlap=TOKEN_OVERLAP,
                       tokenizer=get_tokenizer(SETTINGS), context=context or path)
    for index, part in enumerate(parts):
        _counter[0] += 1
        tokens = token_count(part)
        _chunks.append({
            "id": f"chunk_{_counter[0]:05d}",
            "path": f"{path} [part {index + 1}/{len(parts)}]" if len(parts) > 1 else path,
            "identifier": identifier,
            "text": part,
            "char_count": len(part),
            "token_count": tokens,
            "token_est": tokens,  # Retained for existing JSONL consumers; now exact.
        })

def make_prefix(path: str, heading: str) -> str:
    h = f" [{heading}]" if heading else ""
    return f"{path}{h}: "

def chunk_node(elem, path: str):
    """
    Recursively chunk a USLM element.
    - Fits in TOKEN_LIMIT  → emit as single chunk.
    - Too large            → recurse into structural children.
    - Leaf still too large → sentence-split with overlap.
    """
    tag        = strip_ns(elem.tag)
    label      = get_label(elem)
    heading    = get_heading(elem)
    identifier = get_identifier(elem)
    node_path  = (f"{path} {label}".strip() if label and label != tag else path)
    prefix     = make_prefix(node_path, heading)
    full_text  = prefix + get_text(elem)

    if token_count(full_text) <= TOKEN_LIMIT:
        emit(node_path, identifier, full_text)
        return

    current_idx = HIERARCHY.index(tag) if tag in HIERARCHY else -1
    next_tags   = HIERARCHY[current_idx + 1:] if current_idx >= 0 else HIERARCHY

    structural_children = [
        c for c in elem
        if strip_ns(c.tag) in next_tags
    ]

    if structural_children:
        for child in structural_children:
            chunk_node(child, node_path)
    else:
        emit(node_path, identifier, full_text, context=prefix)

# ── SECTION DISPATCHER ────────────────────────────────────────────────────
def process_section(sec):
    identifier = get_identifier(sec)
    sec_num    = identifier.split("/s")[-1] if "/s" in identifier else get_label(sec)
    sec_path   = f"26 U.S.C. § {sec_num}"
    chunk_node(sec, sec_path)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    src = Path(FILE)
    if not src.exists():
        sys.exit(f"File not found: {FILE}")

    print(f"Parsing {FILE} ...")
    root = ET.parse(src).getroot()

    sections = root.findall(f".//{ns('section')}")
    if not sections:
        sections = root.findall(".//section")
    print(f"Found {len(sections):,} sections. Chunking ...")

    for sec in sections:
        process_section(sec)

    out_path = Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in _chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [c["token_count"] for c in _chunks]
    over       = sum(1 for t in token_counts if t > TOKEN_LIMIT)
    avg        = sum(token_counts) / len(token_counts) if token_counts else 0

    print(f"\nDone.")
    print(f"  Total chunks        : {len(_chunks):,}")
    print(f"  Chunks > {TOKEN_LIMIT} tokens  : {over:,}")
    print(f"  Avg tokens          : {avg:.0f}")
    print(f"  Max tokens          : {max(token_counts, default=0)}")
    print(f"  Min tokens          : {min(token_counts, default=0)}")
    print(f"  Output              : {out_path.resolve()}")

    print("\n── Sample chunks ──────────────────────────────────────────")
    for c in _chunks[:3]:
        print(f"\n  id         : {c['id']}")
        print(f"  path       : {c['path']}")
        print(f"  tokens     : {c['token_count']}")
        print(f"  text       : {c['text'][:280].replace(chr(10), ' ')} ...")

if __name__ == "__main__":
    main()
